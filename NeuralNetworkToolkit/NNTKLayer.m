//
//  NNTKLayer.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import "NNTKLayer.h"
#import <Accelerate/Accelerate.h>

@implementation NNTKLayer

- (instancetype)initWithInputSize:(NSUInteger)inputSize outputSize:(NSUInteger)outputSize activationFunction:(id<NNTKActivationFunction>)activationFunction {
    self = [super init];
    if (self) {
        self.inputSize = inputSize;
        self.outputSize = outputSize;
        
        // Initialize weights and biases with garbage values
        NSUInteger weightsSize = inputSize * outputSize * sizeof(float);
        self.weights = malloc(weightsSize);
        
        NSUInteger biasesSize = outputSize * sizeof(float);
        self.biases = malloc(biasesSize);
        
        self.activationFunction = activationFunction;
    }
    return self;
}

- (NSData *)forward:(NSData *)input {
    // Convert input NSData to float array
    const float *inputBuffer = input.bytes;
    
    // Allocate memory for the output buffer
    float *outputBuffer = malloc(self.outputSize * sizeof(float));
    
    // Compute the weighted sum of inputs and biases
    float weightedSum[self.outputSize];
    memset(weightedSum, 0, self.outputSize * sizeof(float));
    vDSP_mmul(self.weights, 1, inputBuffer, 1, weightedSum, 1, self.outputSize, 1, self.inputSize);
    vDSP_vadd(weightedSum, 1, self.biases, 1, outputBuffer, 1, self.outputSize);
    
    // Apply the activation function to each element in the output buffer
    [self.activationFunction compute:outputBuffer length:self.outputSize];
    
    // Convert output float array to NSData without copying the output buffer
    NSData *outputData = [NSData dataWithBytesNoCopy:outputBuffer length:self.outputSize * sizeof(float)];
    
    return outputData;
}

- (void)deallocate {
    free(self.weights);
    free(self.biases);
}

@end
