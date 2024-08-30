This is the multi-property model based on CrabNet. The idea is to train encoder simultaniously on several property datasets (here there are 2 of them, but it can be extended). 

The network consist of common transformer encoder and separate projection heads. The loss is sum of losses from both heads (the loss from the head is evaluated, if corresponding entry has the data for this head) 

Each datapoint has to have at least on of the properties
