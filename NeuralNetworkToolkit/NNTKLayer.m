//
//  NNTKLayer.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import "NNTKLayer.h"
#import <Accelerate/Accelerate.h>

@implementation NNTKLayer

- (instancetype)initWithInputSize:(NSUInteger)inputSize outputSize:(NSUInteger)outputSize {
    self = [super init];
    if (self) {
        self.inputSize = inputSize;
        self.outputSize = outputSize;
        
        // Initialize weights and biases with random values
        NSUInteger weightsSize = inputSize * outputSize * sizeof(float);
        self.weights = [self randomDataOfSize:weightsSize];
        
        NSUInteger biasesSize = outputSize * sizeof(float);
        self.biases = [self randomDataOfSize:biasesSize];
    }
    return self;
}

- (NSData *)forward:(NSData *)input {
    // Make sure input size matches layer input size
    if (input.length != self.inputSize * sizeof(float)) {
        [NSException raise:@"Invalid input size" format:@"Expected input size %lu, but got %lu", (unsigned long)self.inputSize, input.length / sizeof(float)];
    }
    
    // Allocate memory for output
    float *outputBuffer = malloc(self.outputSize * sizeof(float));
    
    // Compute matrix multiplication
    vDSP_mmul((float *)self.weights.bytes, 1, (float *)input.bytes, 1, outputBuffer, 1, self.outputSize, 1, self.inputSize);
    
    // Add biases to output
    vDSP_vadd(outputBuffer, 1, (float *)self.biases.bytes, 1, outputBuffer, 1, self.outputSize);
    
    // Apply ReLU activation function
    vDSP_vmax(outputBuffer, 1, &zero, outputBuffer, 1, self.outputSize);
    
    // Convert output buffer to NSData object
    NSData *output = [NSData dataWithBytesNoCopy:outputBuffer length:self.outputSize * sizeof(float)];
    outputBuffer = NULL;
    
    return output;
}

- (NSData *)randomDataOfSize:(NSUInteger)size {
    void *buffer = malloc(size);
    arc4random_buf(buffer, size);
    return [NSData dataWithBytesNoCopy:buffer length:size freeWhenDone:YES];
}

@end
