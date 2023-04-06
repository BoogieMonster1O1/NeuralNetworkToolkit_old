//
//  NNTKNeuralNetwork.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import <Cocoa/Cocoa.h>
#import "ActivationFunctions/NNTKActivationFunction.h"
#import "NNTKLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface NNTKNeuralNetwork : NSObject

@property (nonatomic,strong) NSMutableArray<NNTKLayer *> * layers;
@property (nonatomic,assign) NSUInteger inputDimension;
@property (nonatomic,assign) NSUInteger outputDimension;

- (instancetype)initWithInputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension outputActivation:(id<NNTKActivationFunction>)outputActivation;

// Adds a hidden layer before the output layer. Also recreates the output layer.
- (void)addHiddenLayer:(NSUInteger)outputDimension activationFunction:(id<NNTKActivationFunction>)activationFunction;

- (NSData *)forward:(NSData *)input;

- (void)deallocateLayers;

@end

NS_ASSUME_NONNULL_END
