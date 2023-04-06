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
@property (nonatomic, strong) NSData *weights;
@property (nonatomic, strong) NSData *biases;
@property (nonatomic, strong, non) id<NNTKActivationFunction> activationFunction;

- (instancetype)initWithInputSize:(NSUInteger)inputSize outputSize:(NSUInteger)outputSize activationFunction:(id<NNTKActivationFunction>)activationFunction;

- (NSData *)forward:(NSData *)input;

@end

NS_ASSUME_NONNULL_END
