# CoCoaH
Hate Speech Detection using Context-Comment Relationship   
Improving Hate Speech Detection by Learning Relationships between Contexts and Comments

## Abstract
Hate speech on online social media platforms is a growing problem, highlighting the need for effective hate speech detection and mitigation strategies. In this paper, we aim to address these issues by detecting hate speech from comment data. Due to the difficulty in understanding their meaning from comments alone, we propose the Contextual hate speech detection method, CoCoaH(Context-Comment aware Hate Speech Detection). CoCoaH utilizes a title that comprehensively contains the content of the text as contextual information to solve the problem, as it is difficult to detect hate speech from comments alone. The proposed method aims to better capture the meaning of the comments by considering the relationship between the title, which is the context, and the comments. We propose an auxiliary task called Context-Comment Pair Prediction to facilitate learning of these relationships. Using a single model without utilizing ensemble methods, our experiments demonstrate that CoCoaH is effective in detecting hate speech across various datasets and outperforms previous methods.

## Model
![figure3](https://github.com/na2na8/CoCoaH/assets/32005272/7ee10659-05b6-470e-b94c-9350bfcc11aa)<br>
Our goal is to detect hate speech in comments using titles that comprehensively represent the context. We introduce CoCoaH, a model designed to effectively detect hate speech by understanding the relationships between comments and titles. It detects hate speech through contextual hate speech detection as a main task. To enhance the understanding of relationships between titles and comments, we propose an auxiliary task called context-comment pair prediction. This task predicts whether a title and comment share the same topic. CoCoaH trains both the main and auxiliary tasks simultaneously.


## Experiments
![image](https://github.com/na2na8/CoCoaH/assets/32005272/f95d3c0d-d2e4-423e-8600-6f3eb3638973)<br>

![image](https://github.com/na2na8/CoCoaH/assets/32005272/f3477823-649d-4b78-89fb-26eb15f843f2)
