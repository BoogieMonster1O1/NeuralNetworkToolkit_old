//
//  NNTKActivationFunction.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@protocol NNTKActivationFunction <NSObject>

- (float)compute:(float)x;

- (float)derivative:(float)x;

- (float)derivativeOfOutput:(float)output;

- (void)compute:(float*)inputBuffer length:(NSUInteger)length;

@end

NS_ASSUME_NONNULL_END
