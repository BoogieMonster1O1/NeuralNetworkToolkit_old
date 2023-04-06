//
//  NNTKNeuralNetwork.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import "NNTKNeuralNetwork.h"
#import "ActivationFunctions/NNTKActivationFunction.h"

@implementation NNTKNeuralNetwork

- (instancetype)initWithInputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension outputActivation:(id<NNTKActivationFunction>)outputActivation {
    self = [super init];
    if (self) {
        self.layers = [[NSMutableArray alloc] init];
        self.inputDimension = inputDimension;
        self.outputDimension = outputDimension;
        NNTKLayer* layer = [[NNTKLayer alloc] initWithInputSize:inputDimension outputSize:outputDimension activationFunction:outputActivation];
        [self.layers addObject:layer];
    }
    return self;
}

- (void)addHiddenLayer:(NSUInteger)outputDimension activationFunction:(id<NNTKActivationFunction>)activationFunction {
    if (self.layers.count == 0) {
        NSLog(@"Error: No layers found in neural network.");
        return;
    }
    NNTKLayer *outputLayer = [self.layers lastObject];
    NNTKLayer *hiddenLayer = [[NNTKLayer alloc] initWithInputSize:outputLayer.inputSize outputSize:outputDimension activationFunction:activationFunction];
    NNTKLayer *newOutputLayer = [[NNTKLayer alloc] initWithInputSize:outputDimension outputSize:outputLayer.outputSize activationFunction:activationFunction];
    [self.layers removeLastObject];
    [self.layers addObject:hiddenLayer];
    [self.layers addObject:newOutputLayer];
}

- (NSData *)forward:(NSData *)input {
    NSData *data = input;
    for (NNTKLayer *layer in self.layers) {
        data = [layer forward:data];
    }
    return data;
}

- (void)deallocateLayers {
    for (NNTKLayer *layer in self.layers) {
        [layer deallocate];
    }
}

@end
