// Modified by the GIMP LaMa inpainting fork of OxiONNX (2026-09): registers the add_batchnorm fusion module.
// See ../../../../MODIFICATIONS.md for the full change list and rationale.

//! Conv-related fusion passes:
//! - Conv + BatchNorm folding (weight baking)
//! - BatchNorm through a residual Add of two Convs (FFC-style block)
//! - Conv + Relu / Clip activation fusion
//! - Conv + Clip(0,6)  Conv with ReLU6 activation
//! - Conv + Add + ReLU fusion (ResNet residual block pattern)
//! - Standalone BatchNorm folding (Mul + Add replacement)

mod add_batchnorm;
mod add_relu;
mod batchnorm;
mod relu;
mod relu6;

pub use add_batchnorm::fuse_add_batchnorm;
pub use add_relu::fuse_conv_add_relu;
pub use batchnorm::{fold_batch_norm_inference, fuse_conv_batchnorm};
pub use relu::fuse_conv_relu;
pub use relu6::fuse_conv_clip_to_conv_relu6;

#[cfg(test)]
mod tests;
