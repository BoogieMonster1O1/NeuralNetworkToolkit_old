//
//  NNTKLayer.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import <Cocoa/Cocoa.h>
#import "NNTKActivationFunction.h"

NS_ASSUME_NONNULL_BEGIN

@interface NNTKLayer : NSObject

@property (nonatomic, assign) NSUInteger inputSize;
@property (nonatomic, assign) NSUInteger outputSize;
@property (nonatomic, assign) float *weights;
@property (nonatomic, assign) float *biases;
@property (nonatomic, strong) id<NNTKActivationFunction> activationFunction;

- (instancetype)initWithInputSize:(NSUInteger)inputSize outputSize:(NSUInteger)outputSize activationFunction:(id<NNTKActivationFunction>)activationFunction;

- (NSData *)forward:(NSData *)input;

- (void)deallocate;

@end

NS_ASSUME_NONNULL_END
