![](gifs/coin_collect_person2.gif)

See this [accompanying article](https://ctawong.com/articles/roblox_bot.html) for implementation details.

# RobloxBot
Bots that perform policy actions based on Roblox pixel video input.

Bots for `Murder Mystery 2` to collect coins are available.

## System requirement
Windows systems only. GPU is required for  Yolov5-based detector otherwise it will be unbearably slow.

## Install
Packages and versions I used are listed in `requirements.txt`. 

## Software structure
Game-specific codes are under `bots/` directory, with one folder per bot.

Generic functions that are shared by all bots are outside of the bots directory.
- `roblox/` directory contains API to capture screens, emulate keyboard/mouse controls and other helper functions.
- `yolo/` directory contains Yolov5 and wrapper functions.

## Bots available
### Yolo-based object detector
The bot is located at `bots/mm2_yolo_coin_collector`. Run by command

```
cd bots/mm2_yolo_coin_collector/
python main.py
```

All pretrained weights are under `weights/` directory.

The bot either walks to the coins, if detected, or perform random motions.

### Pretrained weights
#### yolo_coin_m_v3.pt
This network is trained to detect standard golden colored coins.  You can run this bot by
```
python main.py --weights weights/yolo_coin_m_v3.pt
```
 under `bots/mm2_yolo_coin_collector` directory.

#### yolo_coin_person_m_v2.pt
This network is trained to detect 
- standard gold coins
- 2020 Christmas coins
- Other players

When other players are detected, the bot will try to move away from them to avoid getting murdered.

You can run this bot by
```
cd bots/mm2_yolo_coin_collector
python main.py --weights weights/yolo_coin_person_m_v2.pt
```

## Training your own Yolo detector
See [accompanying article](https://ctawong.com/articles/roblox_bot.html#appendix-train-your-own-bot).

## Additional Example
### Video of Yolo bot detecting coins only.
![](gifs/coin_collect1.gif)
![](gifs/coin_collect2.gif)

### Video of Yolo bot detecting both coins and other players.

![](gifs/coin_collect_person2.gif)

