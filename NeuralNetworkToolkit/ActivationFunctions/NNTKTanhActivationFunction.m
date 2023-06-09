//
//  NNTKTanhActivationFunction.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/6/23.
//

#import "NNTKTanhActivationFunction.h"

@implementation NNTKTanhActivationFunction

- (float)compute:(float)x {
    return tanh(x);
}

- (float)derivative:(float)x {
    double output = [self compute:x];
    return 1 - output * output;
}

- (float)derivativeOfOutput:(float)output {
    return 1 - output * output;
}

- (void)compute:(float*)inputBuffer length:(NSUInteger)length {
    for (int i = 0; i < length; i++) {
        inputBuffer[i] = [self compute:inputBuffer[i]];
    }
}

@end
