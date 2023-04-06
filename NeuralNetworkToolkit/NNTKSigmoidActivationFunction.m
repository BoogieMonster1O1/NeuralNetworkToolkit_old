//
//  NNTKSigmoidActivationFunction.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import "NNTKSigmoidActivationFunction.h"

@implementation NNTKSigmoidActivationFunction

- (double)activate:(double)x {
    return 1.0 / (1.0 + exp(-x));
}

- (double)derivative:(double)x {
    double s = [self activate:x];
    return s * (1.0 - s);
}

- (float)derivativeOfOutput:(float)output {
    return output * (1.0f - output);
}

@end
