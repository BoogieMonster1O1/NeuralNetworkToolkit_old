//
//  NNTKReLUActivationFunction.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import "NNTKReLUActivationFunction.h"

@implementation NNTKReLUActivationFunction

- (float)compute:(float)x {
    return x > 0 ? x : 0;
}

- (float)derivative:(float)x {
    return x > 0 ? 1 : 0;
}

- (float)derivativeOfOutput:(float)output {
    return output > 0 ? 1 : 0;
}

- (void)compute:(float*)inputBuffer length:(int)length {
    for (int i = 0; i < length; i++) {
        inputBuffer[i] = [self compute:inputBuffer[i]];
    }
}

@end
