# ICPD
===============================

Repository for developing the interactive algorithm for change point detection (CPD) task. More information can be found in the paper **A Semi-supervised Interactive Algorithm for Change Point Detection**.


The authors of this paper are:

- [Zhenxiang Cao](https://www.esat.kuleuven.be/stadius/person.php?id=2380) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
- [Nick Seeuws](https://www.esat.kuleuven.be/stadius/person.php?id=2318) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
- [Maarten De Vos](https://www.esat.kuleuven.be/stadius/person.php?id=203) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven and Dept. Development and Regeneration, KU Leuven)
- [Alexander Bertrand](https://www.esat.kuleuven.be/stadius/person.php?id=331) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)

All authors are affiliated to [LEUVEN.AI - KU Leuven Institute for AI](https://ai.kuleuven.be). 

Abstract
------------
The goal of change point detection (CPD) is to localize abrupt changes in the statistics of signals or time series, which reflect the transitions of properties or states in the underlying system. While many statistical and learning-based approaches have been proposed to carry out this task, most state-of-the-art methods still treat this problem in an unsupervised setting. Therefore, there is often a large gap between the algorithm-detected results and the user-expected ones. To bridge this gap, we apply an active-learning strategy to the CPD problem and combine it with the one-class support vector machine (OCSVM) model, resulting in a semi-supervised CPD algorithm that improves itself by asking queries to the end-user. This allows to focus on the detection of those change points that are desired by the user and ignore false positives or irrelevant change points. Our experiment results on diverse simulated and real-life datasets demonstrate that a substantial improvement in detection performance is achieved on both single- and multi-channel time series with a limited amount of queries.

Requirements
------------
This code requires:
**tensorflow**,
**tensorflow-addons**,
**numpy**,
**pandas**,
**scipy**,
**matplotlib**,
**tsfuse**,
**scikit-learn**.

Usage
-----
The main method is implemented in "main.py" and can be run directly from there.

Data Sources
-----
Honeybee Dance: [Link](http://www.sangminoh.org/Research/Entries/2009/1/21_Honeybee_Dance_Dataset.html)
UCI-test: [Link](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
BabyECG: [Link](https://rdrr.io/cran/wavethresh/man/BabyECG.html)
