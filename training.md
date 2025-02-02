# Training

This documents how the game state is represented and how the AI agents were trained.

## How Game State is Represented for each character

We will use an axis aligned radial band around the subject.

Input
```
    .───────.    
  ,'         `.          Unwrap
 ╱             ╲ 0     Radial Band    0                          2pi
;     Char    --:--    ===========>   |--------------------------| 
 ╲             ╱ 2pi
  `.         ,'   
    `───────'     
```

Then bucketize it.

```
0                         2pi          0                          2pi
|--------------------------|  ======>  |-|-|-|-|-|-|-|-|-|-|-|-|-|-|
```

Then, to compute game state, we project the world into these radial buckets.

We track following seven channels. 
```
                0                                              2pi
Targets         |  0|  0|  1|  0|  0|  0|  0|  0|  0|  0|  0|  0|
Bullets         |  0|  0|  0|  0|  1|  0|  0|  1|  0|  0|  0|  0|
Projected Size  |  0|  0|  2|  0|0.1|  0|  0|0.2|  0|  0|  0|  0|
Red/Blue Shift  |  0|  0|  0|  0|  1|  0|  0| -1|  0|  0|  0|  0|
Optical Flow    |  0|  0|  0|  0| -1|  0|  0|  1|  0|  0|  0|  0|
MyPositionX     |0.5|0.5|0.5|0.5|0.5|0.5|0.5|0.5|0.5|0.5|0.5|0.5|
MyPositionY     |0.5|0.5|0.5|0.5|0.5|0.5|0.5|0.5|0.5|0.5|0.5|0.5|
```
Where:
- Target points to the target enemies.
- Bullets points to the the enemy bullets.
- Projected Size is closeness metric. Highest values trump. Largest values is 1.
- Blue/Red Shifts are normalized to roughly -1 to 1.
- Optical Flow is normalized to roughly  -1 to 1.
- MyPositionX and MyPositionY is relative to the world and they are from -1 to 1 where zero is the center of the world.

## How to efficiently compute the game state for each player.

For the time being, there won't be any acceleration structures.
But, we will limit the number of buckets to 36, and exclude things that are too far away.

```
numBuckets    = 36
targetBucket  = floor(arctan(P_opponent - P_self) / 2pi * 36) % 36
bulletBucket  = floor(arctan(P_bullet - P_self) / 2pi * 36) % 36
projectedSize = 1 / (distance_to_obj + 1)^2
shift         = dot(unit(self_to_target), target.velocity / C) where C = 10
opticalFlow   = crossprod(unit(self_to_target), target.velocity / C) where C = 10
mypos_x       = 2*self.x / width - 1
mypos_y       = 2*self.y / height - 1
```

## Outputs

Outputs will include 4 channels of same buckets as inputs.

```
                0                                              2pi
Steer           |  0|  0|  1|  0|  0|  0|  0|  0|  0|  0|  0|  0|
Aim             |  0|  0|  0|  0|  1|  0|  0|  1|  0|  0|  0|  0|
```

Each rows sum up to one.

How does it translate to actions?

- We will pick the strongest activation and steer/aim that way.
- If strongest activation has value lower than certain threshold, we will not move/shoot.

## Model

We will use standard Res-UNet. It is stat of art for these kind of things at this time.

## How to gather training data

We will collect human plays. 
Then also design a deterministic agent, and try to collect more play data.

We will also measure "importance" of sample by measuring how long did the character survive after the sample. 

If character died shortly after taking the move, the important score will be nearly zero.

If character had longevity, it will have weight of 1. Something like linear weighting should be fine.

Once we train the model with the data, we will collect more data by letting 
the AI play against each other to create more data.

We will then add more AI agents into the picture to collect even more data.

Finally, I will play as an expert to create data that would serve as fine-tuning material.



