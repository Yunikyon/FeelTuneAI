# 🎧 FeelTune AI 

<div align="justify"> 
Musics plays a very important role in defining the emotional state of people, and can be an important tool in guiding them towards a desired emotional state. This work suggests a personalised music suggestion system, focusing on users and their emotions, where the main goal is to build and train personalised artificial intelligence models capable of enabling users to transition from an initial to a desired emotion, given their context and previously classified musics. To train the personalised model, each music played to the user during the dataset construction, is placed as the output to be predicted, and each corresponding emotion obtained is placed as the input. This approach assumes that the obtained emotion during the training phase will correspond to the user's desired emotion, while keeping the context and the initial emotion as inputs. The built model is then used to predict which emotional values the ideal music should have to transition the user between emotional states. After music playback, the user's emotion is re-evaluated to verify if the desired emotional state has been achieved. This process is conducted in real-time, updating suggestions based on the user’s analysed emotion. The objective is to either achieve the desired emotion or maintain it if it has already been achieved.
Despite the fact that the application has few songs available and that they do not cover the whole area of the Valence and Arousal graph, we conclude that there is potential in the intersection between music and human-computer interaction, as this is proved by the proposed approach developed and by the results achieved. We consider the performance of the song prediction models as satisfactory and that it is possible to build models capable of transitioning users between emotional states. 
</div>
<br>

**Keywords**: Music · Emotions · Valence · Arousal · Machine Learning

## Characteristics

- Programming Language: python, v.310
- Valence and Arousal Classification Models: usage of [MuVi dataset](https://github.com/AMAAI-Lab/MuVi) and MLP architecture

## Requirements

- keras==2.12.0
- tensorflow==2.12.0
- librosa
- pandas==1.5.3
- mutagen
- optuna
- python-dotenv
- geocoder
- geopy
- PyQt5
- deepface
- pygame
- youtube_dl
- plotly==5.15.0
- kaleido

## Usage Instructions

1. Install all requriments
2. Execute the file ```interface.py```

## Video

https://github.com/Yunikyon/FeelTuneAI/assets/93437355/be96f7d5-2f14-43a3-a9f0-ac12590e8915

