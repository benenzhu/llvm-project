//===- ShapeInferencePass.cpp - Shape Inference ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a Function level pass performing interprocedural
// propagation of array shapes through function specialization.
//
//===----------------------------------------------------------------------===//

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

  

  
  

/// Include the auto-generated definitions for the shape inference interfaces.
 

 
/// The ShapeInferencePass is a pass that performs intra-procedural
/// shape inference.
///
///    Algorithm:
///
///   1) Build a worklist containing all the operations that return a
///      dynamically shaped tensor: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the argument types.
///   3) If the worklist is empty, the algorithm succeeded.
///
 
        
  

     
       

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
       
      
       
        
    

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
      
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
          
         
        

         
      

      // Ask the operation to infer its output shapes.
               
           
        
        
                
                       
         
      
    

    // If the operation worklist isn't empty, this indicates a failure.
      
         
                 
      
    
  

  /// A utility method that returns if the given operation has all of its
  /// operands inferred.
      
        
       
    
  

  /// A utility method that returns if the given operation has a dynamically
  /// shaped result.
      
        
       
    
  

 // namespace

/// Create a Shape Inference pass.
  
   

