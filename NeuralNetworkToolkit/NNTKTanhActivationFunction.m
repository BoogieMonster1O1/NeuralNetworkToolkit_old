//
//  NNTKTanhActivationFunction.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import "NNTKTanhActivationFunction.h"

@implementation NNTKTanhActivationFunction

- (double)compute:(double)x {
    return tanh(x);
}

- (double)derivative:(double)x {
    double output = [self compute:x];
    return 1 - output * output;
}

- (double)derivativeOfOutput:(double)output {
    return 1 - output * output;
}

@end
