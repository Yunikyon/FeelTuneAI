# 🎧 FeelTune AI 

<div align="justify"> 
This work proposes an approach to build a personalized music suggestion model, focusing on users and their emotions. The aim is to explore its feasibility and develop a prototype capable of leading the user to a desired emotion. To this end, some studies associated with this subject were carried out, which were useful in the process of developing the solution, and we also analysed some similar works. Through these, we concluded that although this area has some tools available, these are not suitable for the purpose of having personalized music recommendations for each user individually according to a desired emotion. We then present a proposal and the respective prototype developed with the aspects considered in that proposal. This prototype contains several parts, namely, emotion recognition, songs' classification according to their Valence and Arousal, and user context capture. Despite the fact that the application has few songs available and that they do not cover the whole area of the Valence and Arousal graph, we considered the performance of the song prediction models as satisfactory. It is also necessary to have in mind that not all people express their emotions through facial expressions so, in an ideal situation, we would rely on additional information to assess the person’s emotion. Finally, we conclude that there is potential in the intersection between music and human-computer interaction, where this is proved by the prototype developed and by the results achieved. Some directions for future work are also presented whose purpose is to overcome the limitations encountered. 
</div>
<br>

**Keywords**: Music · Emotions · Valence · Arousal · Machine Learning

## Characteristics

- Programming Language: python, v.310
- Valence and Arousal Classification Models: usage MuVi dataset and SVM architecture

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
2. Execute the file **interface.py**

## Video

https://github.com/Yunikyon/FeelTuneAI/assets/93437355/be96f7d5-2f14-43a3-a9f0-ac12590e8915

