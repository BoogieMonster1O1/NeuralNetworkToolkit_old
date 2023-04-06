//
//  NNTKSigmoidActivationFunction.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import "NNTKSigmoidActivationFunction.h"

@implementation NNTKSigmoidActivationFunction

- (float)compute:(float)x {
    return 1.0f / (1.0f + exp(-x));
}

- (float)derivative:(float)x {
    float s = [self compute:x];
    return s * (1.0f - s);
}

- (float)derivativeOfOutput:(float)output {
    return output * (1.0f - output);
}

- (void)compute:(float*)inputBuffer length:(int)length {
    for (int i = 0; i < length; i++) {
        inputBuffer[i] = [self compute:inputBuffer[i]];
    }
}

@end
