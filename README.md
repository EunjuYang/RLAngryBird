# RLAngryBird

This repository is for reinforcement learning of angry bird (https://aibirds.org/)
We used AngryBird API (https://aibirds.org/basic-gaime-playing-software/getting-started.html)
DDPG algorithm is used and we referenced the ddpg code from (https://github.com/stevenpig/ddpg-aigym)

## How to install

Reference (https://aibirds.org/basic-game-playing-software/getting-started.html) to set up the environment.

- java is required
- Chrome is required
- JPype is required

```javascript
sudo apt-get install google-chrome-stable
```

```javascript
sudo apt-get install python-jpype
```
Please install Chrome extension.
The extension is located in the folder "plugin".

- a. In Chrome, go to Settings -> Tools -> Extensions
- b. Tick "Developer Mode"
- c. Click "Load unpacked extensions ... "
- d. select the "plugin" folder and confirm
- e. Make sure the extension called "Angry Birds interface" is enabled

myAngryBird.jar is not updated because of its file size.
Please re-compile the src/ab/demo/other/ClientActionRobot.java and name it "myAngryBird.jar".
Replace it with original myAngryBird.jar

## How to execute

 - Execute the chrome browser and go to "chrome.angrybirds.com"
 - click "Play" and then click on "Poached Eggs" on the bottom left
 
 - open one terminal and type the command below
 
 ```sh
java -jar ABServer.jar
```


 - open another terminal and type the command below

 ```sh
cd src/{Version You want to Execute}

python RLAgent.py
```


## Three versions

  ### RLBird_RawImage
  ### RLBird_Grid
  ### RLBird_Lowdim
 
