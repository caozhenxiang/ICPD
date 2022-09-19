# ICPD
===============================

Repository for developing the interactive algorithm for change point detection (CPD) task. More information can be found in the paper **An Interactive Algorithm for Change Point Detection**.


The authors of this paper are:

- [Zhenxiang Cao](https://www.esat.kuleuven.be/stadius/person.php?id=2380) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
- [Nick Seeuws](https://www.esat.kuleuven.be/stadius/person.php?id=2318) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
- [Maarten De Vos](https://www.esat.kuleuven.be/stadius/person.php?id=203) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven and Dept. Development and Regeneration, KU Leuven)
- [Alexander Bertrand](https://www.esat.kuleuven.be/stadius/person.php?id=331) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)

All authors are affiliated to [LEUVEN.AI - KU Leuven Institute for AI](https://ai.kuleuven.be). 

Abstract
------------
The change point detection (CPD) task aims to localize the abrupt changes in time series, which reflects the transitions of properties or states in the underlying system. While many statistical and learning-based approaches have been proposed to carry out this task, most state-of-the-art methods still limit this problem to an unsupervised setting. Hence, the massive gap between the algorithm-detected result and the user-expected one is usually not neglectable. To bridge this gap, we apply an active-learning strategy to the CPD problem and combine it with the One-class support vector machine (OCSVM) model. As a result, a semi-supervised CPD algorithm that can interact with the queries from users is proposed in this paper. The exhausted experiment results on diverse simulated and real-life datasets present the benefits that the proposed algorithm is able to take the information about user-desired types of change into account: Impressive detection performance is achieved on both single- and multi-channel time series with a limited amount of queries.

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
