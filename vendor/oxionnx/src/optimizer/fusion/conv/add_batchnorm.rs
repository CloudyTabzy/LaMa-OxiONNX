// Modified by the GIMP LaMa inpainting fork of OxiONNX (2026-09): ADDS the fuse_add_batchnorm pass (BatchNorm through a residual Add of two Convs).
// See ../../../../MODIFICATIONS.md for the full change list and rationale.

//! Fold a BatchNormalization through a residual Add of two convolutions.
//!
//! LaMa's FFC blocks compute `act(bn(convl2l(x) + convg2l(g)))`: two 3x3
//! convolutions whose outputs are summed and then normalised per channel.
//! `fuse_conv_batchnorm` cannot take this one — the BatchNorm's producer is
//! an `Add`, not a `Conv` — and neither can `fuse_conv_add_relu`, which needs
//! the Add and the Relu to be adjacent. This pass closes that gap.
//!
//! Batch normalisation in inference mode is an affine map per output channel,
//!
//! ```text
//! BN(y) = k * y + c        k = scale / sqrt(var + epsilon)
//!                          c = bias - mean * k
//! ```
//!
//! so it distributes over the sum `y = a + b` and can be folded into both
//! convolutions' weights:
//!
//! ```text
//! a' = conv(A weights * k, bias_A * k + c)     b' = conv(B weights * k, bias_B * k)
//! a' + b' = k * (a + b) + c = BN(a + b)
//! ```
//!
//! The whole shift `c` is carried on the first convolution's bias, so the
//! second needs a bias only when it already had one (a zero bias is omitted
//! rather than materialised).
//!
//! # What is replaced, and why that is safe
//!
//! * Both `Conv` nodes are replaced by copies with new weight/bias
//!   initializers; their output tensor names are unchanged, so the Add and
//!   everything else keep referring to the same tensors.
//! * The `Add` node is replaced by a copy that now produces the
//!   **BatchNormalization's** outputs. The Add's own output must therefore be
//!   a droppable intermediate (exactly one consumer — the BatchNorm — and not
//!   a graph output), which is the first condition checked below.
//! * The `BatchNorm` node is removed.
//!
//! Every condition is a *decline*, never a guess: a malformed or shared
//! operand leaves the graph untouched.
//!
//! This is ONNX Runtime's `conv_bn_fusion` reached through a residual: in ORT
//! the same graph shape is fused by `conv_add_fusion` followed by
//! `conv_bn_fusion`; here both branches are Convs, so one pass suffices.

use std::collections::{HashMap, HashSet};

use crate::graph::{Node, OpKind};
use crate::optimizer::graph_utils::{NameAllocator, TensorUsage};
use crate::tensor::Tensor;

/// Fold `BatchNorm(Add(conv_a, conv_b))` into `Add(conv_a', conv_b')`.
///
/// See the module docs for the transformation and its safety conditions.
pub fn fuse_add_batchnorm(
    nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
    output_names: &[String],
) -> Vec<Node> {
    if nodes.len() < 3 {
        return nodes;
    }

    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.clone(), i);
        }
    }

    let usage = TensorUsage::new(&nodes, output_names);
    let mut names = NameAllocator::new(&nodes, weights);

    let mut skip: HashSet<usize> = HashSet::new();
    let mut replacements: HashMap<usize, Node> = HashMap::new();

    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        if !matches!(node.op, OpKind::BatchNorm) || node.inputs.len() < 5 {
            continue;
        }
        // The Add takes over the BatchNorm's outputs, so only *one* output is
        // expressible. Training-mode BatchNorm (running statistics outputs)
        // is not an inference graph; decline rather than drop an output.
        if node.outputs.len() != 1 {
            continue;
        }

        // The Add's output is renamed away, so it must be a droppable
        // intermediate: one consumer (this BatchNorm), not a graph output.
        let add_out = &node.inputs[0];
        if !usage.is_fusable_intermediate(add_out) {
            continue;
        }
        let add_idx = match producer.get(add_out) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&add_idx) || replacements.contains_key(&add_idx) {
            continue;
        }
        let add_node = &nodes[add_idx];
        if !matches!(add_node.op, OpKind::Add) || add_node.inputs.len() != 2 {
            continue;
        }

        // Both branches must be single-consumer Convs without a fused
        // activation (an activation between Conv and Add would reorder).
        let mut conv_idx = [0usize; 2];
        let mut branches_ok = true;
        for (b, operand) in add_node.inputs.iter().enumerate() {
            if !usage.is_fusable_intermediate(operand) {
                branches_ok = false;
                break;
            }
            let idx = match producer.get(operand) {
                Some(&idx) => idx,
                None => {
                    branches_ok = false;
                    break;
                }
            };
            if skip.contains(&idx) || replacements.contains_key(&idx) {
                branches_ok = false;
                break;
            }
            let conv = &nodes[idx];
            if !matches!(conv.op, OpKind::Conv)
                || conv.inputs.len() < 2
                || !conv.attrs.s("activation").is_empty()
            {
                branches_ok = false;
                break;
            }
            conv_idx[b] = idx;
        }
        if !branches_ok {
            continue;
        }

        // BatchNorm parameters: all constant, all per-output-channel.
        let bn_scale = match weights.get(&node.inputs[1]) {
            Some(t) => t.clone(),
            None => continue,
        };
        let bn_bias = match weights.get(&node.inputs[2]) {
            Some(t) => t.clone(),
            None => continue,
        };
        let bn_mean = match weights.get(&node.inputs[3]) {
            Some(t) => t.clone(),
            None => continue,
        };
        let bn_var = match weights.get(&node.inputs[4]) {
            Some(t) => t.clone(),
            None => continue,
        };
        let c_out = bn_scale.data.len();
        if c_out == 0
            || bn_bias.data.len() != c_out
            || bn_mean.data.len() != c_out
            || bn_var.data.len() != c_out
        {
            continue;
        }
        let epsilon = node.attrs.floats.get("epsilon").copied().unwrap_or(1e-5);

        // Per-conv operands must match the same channel count.
        let mut conv_weights = Vec::with_capacity(2);
        let mut conv_biases = Vec::with_capacity(2);
        let mut operands_ok = true;
        for &idx in &conv_idx {
            let conv = &nodes[idx];
            let w = match weights.get(&conv.inputs[1]) {
                Some(t) => t.clone(),
                None => {
                    operands_ok = false;
                    break;
                }
            };
            if w.shape.first() != Some(&c_out) || w.data.len() % c_out != 0 {
                operands_ok = false;
                break;
            }
            // A bias input that exists but is not a constant of the right
            // length is a malformed model: fold nothing rather than guess.
            let b = match conv.inputs.get(2) {
                Some(name) if !name.is_empty() => match weights.get(name) {
                    Some(t) if t.data.len() == c_out => t.data.clone(),
                    _ => {
                        operands_ok = false;
                        break;
                    }
                },
                _ => vec![0.0f32; c_out],
            };
            conv_weights.push(w);
            conv_biases.push(b);
        }
        if !operands_ok {
            continue;
        }

        // Name bases for the synthesised initializers must be available for
        // *both* branches before anything is mutated: a partial rewrite (one
        // convolution rescoped, the other left alone) would corrupt the graph.
        let mut name_bases = Vec::with_capacity(2);
        let mut bases_ok = true;
        for &idx in &conv_idx {
            match nodes[idx].outputs.first() {
                Some(name) if !name.is_empty() => name_bases.push(name.clone()),
                _ => {
                    bases_ok = false;
                    break;
                }
            }
        }
        if !bases_ok {
            continue;
        }

        // factor = scale / sqrt(var + eps);  shift = bias - mean * factor.
        let mut factor = vec![0.0f32; c_out];
        let mut shift = vec![0.0f32; c_out];
        for c in 0..c_out {
            let inv_std = 1.0 / (bn_var.data[c] + epsilon).sqrt();
            factor[c] = bn_scale.data[c] * inv_std;
            shift[c] = bn_bias.data[c] - bn_mean.data[c] * factor[c];
        }

        // Rewrite both convolutions. Their outputs keep their names, so the
        // Add (and every other consumer, of which there are none) is unaware.
        for b in 0..2 {
            let idx = conv_idx[b];
            let conv = &nodes[idx];
            let per_channel = conv_weights[b].data.len() / c_out;
            let mut fused_weight = conv_weights[b].data.clone();
            let mut fused_bias = vec![0.0f32; c_out];
            for c in 0..c_out {
                let start = c * per_channel;
                for w in &mut fused_weight[start..start + per_channel] {
                    *w *= factor[c];
                }
                // The whole shift rides on the first branch.
                fused_bias[c] = conv_biases[b][c] * factor[c];
                if b == 0 {
                    fused_bias[c] += shift[c];
                }
            }

            // Key the generated names on the (spec-unique) Conv output tensor.
            let name_base = &name_bases[b];
            let fused_weight_name = names.allocate(name_base, "_addbn_weight");
            let fused_bias_name = names.allocate(name_base, "_addbn_bias");
            weights.insert(
                fused_weight_name.clone(),
                Tensor::new(fused_weight, conv_weights[b].shape.clone()),
            );
            weights.insert(
                fused_bias_name.clone(),
                Tensor::new(fused_bias, vec![c_out]),
            );

            // Conv inputs are [X, W] or [X, W, B]; the fold always produces a
            // bias (the first branch carries the shift), so create the slot
            // when the convolution did not have one.
            let mut fused_inputs = conv.inputs.clone();
            fused_inputs[1] = fused_weight_name;
            if fused_inputs.len() >= 3 {
                fused_inputs[2] = fused_bias_name;
            } else {
                fused_inputs.push(fused_bias_name);
            }

            replacements.insert(
                idx,
                Node {
                    op: OpKind::Conv,
                    name: format!("{}_addbn", conv.name),
                    inputs: fused_inputs,
                    outputs: conv.outputs.clone(),
                    attrs: conv.attrs.clone(),
                },
            );
        }

        // The Add now carries the normalisation's results.
        replacements.insert(
            add_idx,
            Node {
                op: OpKind::Add,
                name: format!("{}_addbn", add_node.name),
                inputs: add_node.inputs.clone(),
                outputs: node.outputs.clone(),
                attrs: add_node.attrs.clone(),
            },
        );
        skip.insert(i);
    }

    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}
