//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "toy/Dialect.h"
using namespace mlir;
using namespace toy;
 
namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
/// /* 目前好像还没有 */
#include "ToyCombine.inc"
} // namespace
 
/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
/* Fold transpose(transpose(x)) -> x */
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
     
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
    /* 看下 OpRewritePattern 是啥？  */
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  llvm::LogicalResult
    /* 看下 matchAndRewrite 是啥？ */
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();     
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>(); 

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    /* 直接使用 InputOp 的 getOperand 来替换 应该是直接省去了两个transpose的吧*/
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
}; 


/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
/// /* TODO 看下这个直接调用到哪里了? 还是 op 定义直接出来的 */
/// /* let hasCanonicalizer = 1; */ 直接出来的。。。
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern>(context);
}
                                              
   
              


