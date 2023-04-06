//
//  NNTKActivationFunction.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@protocol NNTKActivationFunction <NSObject>

- (double)compute:(double)x;

- (double)derivative:(double)x;

- (double)derivativeOfOutput:(double)output;

@end

NS_ASSUME_NONNULL_END
