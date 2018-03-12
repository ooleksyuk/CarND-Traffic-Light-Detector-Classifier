## Traffic light Classifier

The project is aimed at classifying traffic light on the incoming picture either from Simulator or from Carla.

I've used two different models for Simulator and for Carla.

Simulator model summary:
```bash
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 64, 32, 32)        896       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 32, 16, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 32, 16, 32)        9248      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 16, 8, 32)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 4096)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 32776     
_________________________________________________________________
dense_4 (Dense)              (None, 4)                 36        
=================================================================
Total params: 42,956
Trainable params: 42,956
Non-trainable params: 0
_________________________________________________________________
```

Carla model summary:
```bash
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_2 (Conv2D)            (None, 64, 32, 32)        896       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 16, 32)        0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 16384)             0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 131080    
_________________________________________________________________
dense_4 (Dense)              (None, 4)                 36        
=================================================================
Total params: 132,012
Trainable params: 132,012
Non-trainable params: 0
_________________________________________________________________
```
I have trained both models with the following parameters:
```python
epochs=30, 
validation_split=0.1, 
shuffle=True
```
For Simulator I has more data and used `batch_size=128` vs data from Carla was insufficient and I had to improve training by increasing batch side to 256 `batch_size=256`

Here is an example of light color prediction for Carla and Simulator:
![Carla](./red_carla.png)

![Simulator](./green_sim.png)