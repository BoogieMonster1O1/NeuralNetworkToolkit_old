//
//  NNTKReLUActivationFunction.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import "NNTKReLUActivationFunction.h"

@implementation NNTKReLUActivationFunction

- (double)compute:(double)x {
    return x > 0 ? x : 0;
}

- (double)derivative:(double)x {
    return x > 0 ? 1 : 0;
}

- (double)derivativeOfOutput:(double)output {
    return output > 0 ? 1 : 0;
}

@end
