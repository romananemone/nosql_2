# Семантичний пошук за науковими статтями

10 000 статей з arXiv, ембеддинги `allenai/specter2_base`, індекс у Pinecone, пошук з фільтрами, chunking і гібридний пошук BM25 + вектор через RRF.

## Як запустити

```bash
py -3.12 -m venv .venv            # потрібен Python 3.11 або новіший
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env              # вписати PINECONE_API_KEY

kaggle datasets download -d Cornell-University/arxiv
python -m zipfile -e arxiv.zip .

python scripts/01_prepare_data.py
python scripts/02_embed.py        # ~18 хв на CPU
python scripts/03_load_to_pinecone.py
python scripts/04_search.py
python scripts/05_chunking.py
python scripts/06_hybrid_search.py
```

`.env`, `data/`, `embeddings/` і датасет у `.gitignore`.

---

## Частина 1. Дані і вибір інструментів

Перші 10 000 рядків файлу це тільки квітень-травень 2007 і 10 статей `cs.LG`, фільтри з Частини 3 на такому не працюють. Тому `01_prepare_data.py` читає весь файл і випадково бере статті з `seed=42`: 60% `cs.*`, 40% решта. Перезапуск дає той самий parquet.

<details><summary>Вивід 01_prepare_data.py</summary>

```
Переглянуто статей: cs.* = 855015, інші = 2397437
Завантажено статей: 10000

Розподіл за категоріями (топ-10):
category
cs.CV       1179
cs.LG       1062
cs.CL        599
cs.AI        392
cs.RO        329
cs.IT        319
hep-ph       247
quant-ph     236
cs.CR        227
hep-th       193
Name: count, dtype: int64

Розподіл за роками:
year
1991       1
1992       7
1993      11
1994      24
1995      23
1996      35
1997      31
1998      39
1999      56
2000      63
2001      57
2002      71
2003      69
2004      82
2005      82
2006      86
2007      99
2008      95
2009     138
2010     152
2011     157
2012     194
2013     218
2014     248
2015     251
2016     306
2017     362
2018     426
2019     476
2020     575
2021     595
2022     674
2023     780
2024     948
2025    1181
2026    1388
Name: count, dtype: int64

Приклад запису:
{'id': '0704.0149', 'title': 'Construction of initial data for 3+1 numerical relativity', 'abstract': 'This lecture is devoted to the problem of computing initial data for the Cauchy problem of 3+1 general relativity. The main task is to solve the constraint equations. The conformal technique, introduced by Lichnerowicz and enhanced by York, is presented. Two standard methods, the conformal transverse-traceless one and the conformal thin sandwich, are discussed and illustrated by some simple examples. Finally a short review regarding initial data for binary systems (black holes and neutron stars) is given.', 'authors': 'Gourgoulhon Eric', 'year': 2007, 'category': 'gr-qc'}

Збережено в data/arxiv_subset.parquet
```
</details>

### 1. Pinecone, Qdrant і Chroma

Pinecone це тільки хмарний сервіс із закритим кодом: нічого не треба адмініструвати, масштабування на провайдері, але ти прив'язаний до нього, платиш за використання, і кожен запит іде мережею. Qdrant відкритий (Apache 2.0), на Rust, його можна підняти у себе в Docker або взяти їхню хмару; він швидкий, добре фільтрує і вміє квантизацію. Chroma теж відкрита (Apache 2.0), найпростіша, працює прямо в Python-процесі, але на мільйонах векторів слабша за інші дві. Я б брав Pinecone, коли не хочеться тримати свою інфраструктуру або для навчального проєкту, Qdrant, коли дані не можна віддавати назовні чи важлива ціна на великих обсягах, а Chroma для прототипів і локальних експериментів.

### 2. Чому specter2_base, а не all-MiniLM-L6-v2

Картка [specter2_base](https://huggingface.co/allenai/specter2_base): *"SPECTER2 has been trained on over 6M triplets of scientific paper citations"* і *"Given the combination of title and abstract of a scientific paper or a short texual query, the model can be used to generate effective embeddings"*. Тобто модель вчили на наукових статтях, і схожими вважаються статті, які цитують одна одну. Це якраз те, що треба для пошуку статей, і формат `title [SEP] abstract` збігається з нашими даними. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) універсальна, її вчили на ~1 млрд пар звичайних речень, і текст довший за 256 токенів вона обрізає. Вона швидша, але наукові терміни розуміє гірше.

### 3. Метрика схожості з картки

Прямо метрику картка не називає, там тільки *"Nearest Neighbor Search"* як сценарій використання. Це важливо, бо метрику Pinecone задаєш один раз при створенні індексу, і якщо вона не підходить до моделі, сортування буде неправильним. Я нормалізую вектори (`normalize_embeddings=True`), і тоді cosine, dot і L2 сортують однаково, тому індекс створено з `metric="cosine"`. Ще помітив, що картка бере ембеддинг з токена [CLS], а `SentenceTransformer` сам ставить mean pooling. Завдання вимагає саме sentence-transformers, тому я це залишив.

<details><summary>Вивід 02_embed.py</summary>

```
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
Batches: 100%|██████████| 157/157
Кількість текстів:      10000
Розмірність ембеддингу: 768
Норма першого вектора:  1.000000
Збережено в embeddings/embeddings.npy
```
</details>

### Чому для нормалізованих векторів cosine дорівнює dot product

$\cos(a, b) = \frac{a \cdot b}{\|a\|\,\|b\|}$. Якщо $\|a\| = \|b\| = 1$, знаменник дорівнює 1, і $\cos(a, b) = a \cdot b$.

---

## Частина 2. Завантаження в Pinecone

Індекс `arxiv-papers` (768, cosine, aws us-east-1), батчі по 200, id `paper_<номер рядка>`, метадані `arxiv_id`, `title`, `abstract[:500]`, `authors[:200]`, `year`, `category`.

<details><summary>Вивід 03_load_to_pinecone.py</summary>

```
Індекс 'arxiv-papers' вже існує, використовуємо його
Статей: 10000, ембеддинги: (10000, 768)

Векторів в індексі: 10000
Розмірність: 768
{'dimension': 768,
 'index_fullness': 0.0,
 'namespaces': {'': {'vector_count': 10000}},
 'total_vector_count': 10000}
```
</details>

---

## Частина 3. Пошукові запити

<details><summary>Вивід 04_search.py</summary>

```
Норма вектора запиту: 1.000000


Семантичний пошук: 'teaching machines to recognize objects in pictures'
1. [0.8615] Vision Transformers: State of the Art and Research Challenges
   cs.CV | 2022 | arXiv:2207.03041
   Transformers have achieved great success in natural language processing. Due to the powerful capability of self-attention mechanism in transformers, researchers develop the vision transformers for a v...
2. [0.8612] Visual Transfer Learning: Informal Introduction and Literature Overview
   cs.CV | 2012 | arXiv:1211.1127
   Transfer learning techniques are important to handle small training sets and to allow for quick generalization even from only a few examples. The following paper is the introduction as well as the lit...
3. [0.8521] Learning Explanations from Language Data
   cs.CL | 2018 | arXiv:1808.04127
   PatternAttribution is a recent method, introduced in the vision domain, that explains classifications of deep neural networks. We demonstrate that it also generates meaningful interpretations in the l...
4. [0.8504] Machine-learnable Sets
   cs.LG | 2026 | arXiv:2606.28947
   In this study we present a formal definition of large discrete sets having, informally, three properties: their elements are easily recognized, easily generated, and the latter tasks are easily learne...
5. [0.8499] Morphological Image Analysis and Feature Extraction for Reasoning with AI-based Defect Detection and Classification Models
   cs.CV | 2023 | arXiv:2307.11643
   As the use of artificial intelligent (AI) models becomes more prevalent in industries such as engineering and manufacturing, it is essential that these models provide transparent reasoning behind thei...


A: 'agents learning to act through trial and error and rewards' | category = cs.LG, year >= 2022
1. [0.8874] Weighted Maximum Entropy Inverse Reinforcement Learning
   cs.LG | 2022 | arXiv:2208.09611
   We study inverse reinforcement learning (IRL) and imitation learning (IM), the problems of recovering a reward or policy function from expert's demonstrated trajectories. We propose a new way to impro...
2. [0.8754] Towards Adapting Reinforcement Learning Agents to New Tasks: Insights from Q-Values
   cs.LG | 2024 | arXiv:2407.10335
   While contemporary reinforcement learning research and applications have embraced policy gradient methods as the panacea of solving learning problems, value-based methods can still be useful in many d...
3. [0.8754] Backward Learning for Goal-Conditioned Policies
   cs.LG | 2023 | arXiv:2312.05044
   Can we learn policies in reinforcement learning without rewards? Can we learn a policy just by trying to reach a goal state? We answer these questions positively by proposing a multi-step procedure th...
4. [0.8753] Which Experiences Are Influential for Your Agent? Policy Iteration with Turn-over Dropout
   cs.LG | 2023 | arXiv:2301.11168
   In reinforcement learning (RL) with experience replay, experiences stored in a replay buffer influence the RL agent's performance. Information about the influence is valuable for various purposes, inc...
5. [0.8709] LLM-Driven Stationarity-Aware Expert Demonstrations for Multi-Agent Reinforcement Learning in Mobile Systems
   cs.LG | 2025 | arXiv:2511.19368
   Multi-agent reinforcement learning (MARL) has been increasingly adopted in many real-world applications. While MARL enables decentralized deployment on resource-constrained edge devices, it suffers fr...


B: 'agents learning to act through trial and error and rewards' | year < 2015, будь-яка категорія
1. [0.8495] Feature Markov Decision Processes
   cs.AI | 2008 | arXiv:0812.4580
   General purpose intelligent learning agents cycle through (complex,non-MDP) sequences of observations, actions, and rewards. On the other hand, reinforcement learning is well-developed for small finit...
2. [0.8425] Taking a shower in Youth Hostels: risks and delights of heterogeneity
   physics.soc-ph | 2008 | arXiv:0801.1573
   Tuning one's shower in some hotels may turn into a challenging coordination game with imperfect information. The temperature sensitivity increases with the number of agents, making the problem possibl...
3. [0.8402] Simulating Non Stationary Operators in Search Algorithms
   cs.AI | 2014 | arXiv:1409.1686
   In this paper, we propose a model for simulating search operators whose behaviour often changes continuously during the search. In these scenarios, the performance of the operators decreases when they...
4. [0.8373] Optimizing Dialogue Management with Reinforcement Learning: Experiments with the NJFun System
   cs.LG | 2011 | arXiv:1106.0676
   Designing the dialogue policy of a spoken dialogue system involves many nontrivial choices. This paper presents a reinforcement learning approach for automatically optimizing a dialogue policy, which ...
5. [0.8347] Visual Transfer Learning: Informal Introduction and Literature Overview
   cs.CV | 2012 | arXiv:1211.1127
   Transfer learning techniques are important to handle small training sets and to allow for quick generalization even from only a few examples. The following paper is the introduction as well as the lit...


Без фільтрів: 'agents learning to act through trial and error and rewards'
1. [0.8954] Guiding Robot Exploration in Reinforcement Learning via Automated Planning
   cs.RO | 2020 | arXiv:2004.11456
   Reinforcement learning (RL) enables an agent to learn from trial-and-error experiences toward achieving long-term goals; automated planning aims to compute plans for accomplishing tasks using action k...
2. [0.8931] Complementary reinforcement learning towards explainable agents
   cs.LG | 2019 | arXiv:1901.00188
   Reinforcement learning (RL) algorithms allow agents to learn skills and strategies to perform complex tasks without detailed instructions or expensive labelled training examples. That is, RL agents ca...
3. [0.8874] Weighted Maximum Entropy Inverse Reinforcement Learning
   cs.LG | 2022 | arXiv:2208.09611
   We study inverse reinforcement learning (IRL) and imitation learning (IM), the problems of recovering a reward or policy function from expert's demonstrated trajectories. We propose a new way to impro...
4. [0.8869] Multi-Agent Reinforcement Learning for Unmanned Aerial Vehicle Coordination by Multi-Critic Policy Gradient Optimization
   cs.RO | 2020 | arXiv:2012.15472
   Recent technological progress in the development of Unmanned Aerial Vehicles (UAVs) together with decreasing acquisition costs make the application of drone fleets attractive for a wide variety of tas...
5. [0.8867] Self-adapting Robotic Agents through Online Continual Reinforcement Learning with World Model Feedback
   cs.RO | 2026 | arXiv:2603.04029
   As learning-based robotic controllers are typically trained offline and deployed with fixed parameters, their ability to cope with unforeseen changes during operation is limited. Biologically inspired...

Порівняння видачі:
  спільних A ∩ B:            0
  спільних A ∩ без фільтра:  1
  спільних B ∩ без фільтра:  0
  категорії в B: ['cs.AI', 'cs.CV', 'cs.LG', 'physics.soc-ph']
  роки в A: [2022, 2023, 2023, 2024, 2025]
  роки в B: [2008, 2008, 2011, 2012, 2014]

Локальні ембеддинги: (10000, 768), норми від 1.0000 до 1.0000

Запит: 'teaching machines to recognize objects in pictures'

Cosine similarity (більше = краще):
1. [0.861474] Vision Transformers: State of the Art and Research Challenges
2. [0.861178] Visual Transfer Learning: Informal Introduction and Literature Overview
3. [0.852087] Learning Explanations from Language Data
4. [0.850450] Machine-learnable Sets
5. [0.849942] Morphological Image Analysis and Feature Extraction for Reasoning with AI-based 

Dot product (більше = краще):
1. [0.861474] Vision Transformers: State of the Art and Research Challenges
2. [0.861178] Visual Transfer Learning: Informal Introduction and Literature Overview
3. [0.852087] Learning Explanations from Language Data
4. [0.850450] Machine-learnable Sets
5. [0.849942] Morphological Image Analysis and Feature Extraction for Reasoning with AI-based 

L2 distance (менше = краще):
1. [0.526358] Vision Transformers: State of the Art and Research Challenges
2. [0.526919] Visual Transfer Learning: Informal Introduction and Literature Overview
3. [0.543899] Learning Explanations from Language Data
4. [0.546901] Machine-learnable Sets
5. [0.547829] Morphological Image Analysis and Feature Extraction for Reasoning with AI-based 

Чи збігаються топ-5:
  cosine == dot: True
  cosine == L2:  True
  перевірка L2^2 = 2 - 2*cos для топ-1: 0.277053 vs 0.277053
  Pinecone == локальний cosine: True
```
</details>

**Семантичний пошук.** На *"teaching machines to recognize objects in pictures"* знайшлись Vision Transformers, Visual Transfer Learning і пошук дефектів на зображеннях. Слів із запиту там немає, збіг за змістом.

**Фільтри.** Для них окремий запит про RL. A (`cs.LG`, з 2022) дав 5 справжніх статей про RL. B (до 2015) дав статті 2008-2014 з різних категорій і з нижчим score, бо старих статей про RL у вибірці мало, туди навіть потрапила стаття про налаштування душу в хостелі як гру агентів. Спільних статей у A і B немає. Фільтр завжди повертає 5 результатів, але якщо підходящих документів мало, якість падає.

**Чи збігаються топ-5 для cosine і dot?** Так, і числа однакові до 6 знака, бо вектори нормалізовані і $\cos = a \cdot b$.

**Чи відрізняється L2?** Порядок той самий, змінюється тільки число (відстань, менше краще), бо $\|a-b\|^2 = 2 - 2\cos$. Скрипт це перевіряє: 0.277053 і 0.277053.

**Що було б без нормалізації?** Dot і L2 почали б залежати від довжини векторів, і довгий вектор вигравав би навіть без схожості за змістом. Правильним залишився б тільки cosine.

---

## Частина 4. Chunking

30 найдовших анотацій. Fixed: 50 слів, overlap 10. Semantic: цілі речення, до 60 слів. Окремі індекси `arxiv-chunks-fixed` і `arxiv-chunks-semantic`.

<details><summary>Вивід 05_chunking.py</summary>

```
Обрано 30 статей, довжина анотацій: 301-413 слів

Статистика чанків:
  fixed     чанків:  243 | слів: середнє  47.4, min 13, max 50 | закінчуються посеред речення: 209
  semantic  чанків:  210 | слів: середнє  44.7, min 18, max 77 | закінчуються посеред речення: 0

Приклад для статті: Classification of Complex Networks Based on Topological Properties

  [fixed]
   #0: Complex networks are a powerful modeling tool, allowing the study of countless real-world systems. They have been used in very different domains such as computer science, biology, sociology, management, etc. Authors have been trying to characterize them using various measures such as degree distribution, transitivity or average distance. Their goal
   #1: such as degree distribution, transitivity or average distance. Their goal is to detect certain properties such as the small-world or scale-free properties. Previous works have shown some of these properties are present in many different systems, while others are characteristic of certain types of systems only. However, each one of
   #2: of certain types of systems only. However, each one of these studies generally focuses on a very small number of topological measures and networks. In this work, we aim at using a more systematic approach. We first constitute a dataset of 152 publicly available networks, spanning over 7 different domains.

  [semantic]
   #0: Complex networks are a powerful modeling tool, allowing the study of countless real-world systems. They have been used in very different domains such as computer science, biology, sociology, management, etc. Authors have been trying to characterize them using various measures such as degree distribution, transitivity or average distance.
   #1: Their goal is to detect certain properties such as the small-world or scale-free properties. Previous works have shown some of these properties are present in many different systems, while others are characteristic of certain types of systems only. However, each one of these studies generally focuses on a very small number of topological measures and networks.
   #2: In this work, we aim at using a more systematic approach. We first constitute a dataset of 152 publicly available networks, spanning over 7 different domains. We then process 14 different topological measures to characterize them in the most possible complete way. Finally, we apply standard data mining tools to analyze these data.

Вплив overlap (fixed, size=50), всього слів у текстах: 9388
  overlap= 0: чанків  212, слів у чанках  9388 (x1.00 від тексту)
  overlap=10: чанків  243, слів у чанках 11518 (x1.23 від тексту)
  overlap=20: чанків  305, слів у чанках 14888 (x1.59 від тексту)
  overlap=30: чанків  436, слів у чанках 21568 (x2.30 від тексту)

[fixed] кодуємо 243 чанків

[semantic] кодуємо 210 чанків
Векторів в 'arxiv-chunks-fixed': 243 (очікувалось 243)
Векторів в 'arxiv-chunks-semantic': 210 (очікувалось 210)


Запит: 'searching for habitable Earth-like planets around other stars'

  [fixed]
  1. [0.9034] Transit least-squares survey -- III. A $1.9\,R_\oplus$ transit candidate in the  (чанк #7)
     between about 1 and 20 m/s. We also find the super-Earth-sized transiting planet candidate KOI-456.04 in the habitable zone of this system, which could be the f...
  2. [0.8848] Greening of the Brown Dwarf Desert. EPIC 212036875 b -- a 51 M$_\mathrm{J}$ obje (чанк #4)
     object that resides in the brown dwarf desert. In the mass-density diagram for planets, brown dwarfs and stars, we find that all giant planets and brown dwarfs ...
  3. [0.8762] Greening of the Brown Dwarf Desert. EPIC 212036875 b -- a 51 M$_\mathrm{J}$ obje (чанк #0)
     Our aim is to investigate the nature and formation of brown dwarfs by adding a new well-characterised object to the small sample of less than 20 transiting brow...
  4. [0.8514] Transit least-squares survey -- III. A $1.9\,R_\oplus$ transit candidate in the  (чанк #0)
     The Sun-like star Kepler-160 (KOI-456) has been known to host two transiting planets, Kepler-160 b and c, of which planet c shows substantial transit-timing var...
  5. [0.8506] Habitable-zone Earths at the detection frontier: Measured completeness and false (чанк #5)
     2%). In the field, the 50 known transiting objects of interest (19 confirmed planets, 31 candidates) would all be recovered; a transiting HZ planet of 1.0 (1.5)...

  [semantic]
  1. [0.9015] Transit least-squares survey -- III. A $1.9\,R_\oplus$ transit candidate in the  (чанк #5)
     The expected stellar radial velocity amplitude caused by this new planet ranges between about 1 and 20 m/s. We also find the super-Earth-sized transiting planet...
  2. [0.8789] Greening of the Brown Dwarf Desert. EPIC 212036875 b -- a 51 M$_\mathrm{J}$ obje (чанк #3)
     In the mass-density diagram for planets, brown dwarfs and stars, we find that all giant planets and brown dwarfs follow the same trend from ~0.3 MJ to the turn-...
  3. [0.8707] Greening of the Brown Dwarf Desert. EPIC 212036875 b -- a 51 M$_\mathrm{J}$ obje (чанк #0)
     Our aim is to investigate the nature and formation of brown dwarfs by adding a new well-characterised object to the small sample of less than 20 transiting brow...
  4. [0.8514] Habitable-zone Earths at the detection frontier: Measured completeness and false (чанк #5)
     In the field, the 50 known transiting objects of interest (19 confirmed planets, 31 candidates) would all be recovered; a transiting HZ planet of 1.0 (1.5) Eart...
  5. [0.8473] Greening of the Brown Dwarf Desert. EPIC 212036875 b -- a 51 M$_\mathrm{J}$ obje (чанк #1)
     We combined the K2 photometric data with a series of multi-colour photometric observations, imaging and radial velocity measurements to rule out false positive ...


Запит: 'predicting software bugs with recurrent neural networks'

  [fixed]
  1. [0.9064] Code Failure Prediction and Pattern Extraction using LSTM Networks (чанк #0)
     In this paper, we use a well-known Deep Learning technique called Long Short Term Memory (LSTM) recurrent neural networks to find sessions that are prone to cod...
  2. [0.8825] Code Failure Prediction and Pattern Extraction using LSTM Networks (чанк #1)
     also use LSTM networks to extract telemetry patterns that lead to a specific code failure. For code failure prediction, we treat the telemetry events, sequence ...
  3. [0.8685] Code Failure Prediction and Pattern Extraction using LSTM Networks (чанк #6)
     a code failure from happening. The trained LSTM model shows over 99% accuracy for detecting code failures in the synthetic data. The results from the proposed m...
  4. [0.8639] Code Failure Prediction and Pattern Extraction using LSTM Networks (чанк #2)
     the context of sentiment analysis, respectively. Our proposed method is able to process a large set of data and can automatically handle edge cases in code fail...
  5. [0.8596] Code Failure Prediction and Pattern Extraction using LSTM Networks (чанк #4)
     are the set of events that each of them individually prevents a code failure from happening, even in presence of one or multiple contributor(s). Once the propos...

  [semantic]
  1. [0.9086] Code Failure Prediction and Pattern Extraction using LSTM Networks (чанк #0)
     In this paper, we use a well-known Deep Learning technique called Long Short Term Memory (LSTM) recurrent neural networks to find sessions that are prone to cod...
  2. [0.8633] Code Failure Prediction and Pattern Extraction using LSTM Networks (чанк #1)
     For code failure prediction, we treat the telemetry events, sequence of telemetry events and the outcome of each sequence as words, sentence and sentiment in th...
  3. [0.8591] Code Failure Prediction and Pattern Extraction using LSTM Networks (чанк #3)
     In this paper, contributors are the set of events that cause a code failure, while blockers are the set of events that each of them individually prevents a code...
  4. [0.8516] Code Failure Prediction and Pattern Extraction using LSTM Networks (чанк #4)
     To develop and test our proposed method, we use synthetic (simulated) data in the first step. The synthetic data is generated using a number of rules for code f...
  5. [0.8313] Forecasting How RLHF Responds to Reward-Model Edits Before Training (чанк #0)
     Responses with headers, lists and greater length receive high scores from reward models (RMs), and RLHF policies learn to produce these features even when true ...


Запит: 'fast sorting algorithms for small arrays'

  [fixed]
  1. [0.8659] Engineering Faster Sorters for Small Sets of Items (чанк #2)
     insertion sort to sort small sets of items to provide a more efficient base case sorter. We looked at sorting networks, at how they can improve the speed of sor...
  2. [0.8559] Engineering Faster Sorters for Small Sets of Items (чанк #4)
     we adapted sample sort to Register Sample Sort, to break down those larger sets into sizes that can in turn be sorted by sorting networks. From our experiments ...
  3. [0.8517] Engineering Faster Sorters for Small Sets of Items (чанк #3)
     manner by using conditional moves. Since sorting networks need to be implemented explicitly for each set size, providing networks for larger sizes becomes less ...
  4. [0.8445] Engineering Faster Sorters for Small Sets of Items (чанк #5)
     insertion sort by a factor of at least 1.76 for any array size between six and sixteen, and by a factor of 2.72 on average across all machines and array sizes. ...
  5. [0.8387] Engineering Faster Sorters for Small Sets of Items (чанк #0)
     Sorting a set of items is a task that can be useful by itself or as a building block for more complex operations. That is why a lot of effort has been put into ...

  [semantic]
  1. [0.8655] Engineering Faster Sorters for Small Sets of Items (чанк #2)
     We looked at sorting networks, at how they can improve the speed of sorting few elements, and how to implement them in an efficient manner by using conditional ...
  2. [0.8597] Engineering Faster Sorters for Small Sets of Items (чанк #5)
     When integrating sorting networks as a base case sorter into Quicksort, we achieved far less performance improvements, which is probably due to the networks hav...
  3. [0.8490] Engineering Faster Sorters for Small Sets of Items (чанк #0)
     Sorting a set of items is a task that can be useful by itself or as a building block for more complex operations. That is why a lot of effort has been put into ...
  4. [0.8481] Engineering Faster Sorters for Small Sets of Items (чанк #3)
     To also enable the sorting of slightly larger base cases, we adapted sample sort to Register Sample Sort, to break down those larger sets into sizes that can in...
  5. [0.8420] Engineering Faster Sorters for Small Sets of Items (чанк #6)
     But for x86 machines with a larger L1 instruction cache of 64 KiB or more, we obtained speedups of 12.7% when using sorting networks as a base case sorter in st...


Запит: 'structure of neutron stars and their equations of state'

  [fixed]
  1. [0.8628] Equations of state in the Hartle-Thorne model of neutron stars selecting accepta (чанк #3)
     are confronted with a variety of equations of state (EoS) governing structure of neutron stars in the framework of the Hartle-Thorne theory of rotating neutron ...
  2. [0.8537] Equations of state in the Hartle-Thorne model of neutron stars selecting accepta (чанк #2)
     in the regions related to the resonant points. The most precise variants of the RS model are those combining the relativistic precession frequency relations wit...
  3. [0.8511] Equations of state in the Hartle-Thorne model of neutron stars selecting accepta (чанк #1)
     that the neutron star exterior can be approximated by the Kerr geometry. Strong restrictions of the neutron star parameters M (mass) and a (spin) arise due to f...
  4. [0.8433] Equations of state in the Hartle-Thorne model of neutron stars selecting accepta (чанк #0)
     The Resonant Switch (RS) model of twin high-frequency quasi-periodic oscillations (HF QPOs) observed in neutron star binary systems, based on switch of the twin...
  5. [0.8268] Equations of state in the Hartle-Thorne model of neutron stars selecting accepta (чанк #4)
     the neutron star at 4U 1636-53. It is shown that only two variants of the RS model based on the Kerr approximation are compatible with two EoS applied in the Ha...

  [semantic]
  1. [0.8666] Equations of state in the Hartle-Thorne model of neutron stars selecting accepta (чанк #2)
     Here, the neutron star mass and spin estimates given by the RS model are confronted with a variety of equations of state (EoS) governing structure of neutron st...
  2. [0.8347] Equations of state in the Hartle-Thorne model of neutron stars selecting accepta (чанк #0)
     The Resonant Switch (RS) model of twin high-frequency quasi-periodic oscillations (HF QPOs) observed in neutron star binary systems, based on switch of the twin...
  3. [0.8205] Stellar laboratories: new Ge V and Ge VI oscillator strengths and their validati (чанк #1)
     Many of their lines are identified in spectra of extremely hot, hydrogen-deficient post-AGB stars. A reliable determination of their abundances establishes cruc...
  4. [0.8161] Study of the GeV to TeV morphology of the $\gamma$-Cygni SNR (G78.2+2.1) with MA (чанк #2)
     Since this SNR has the right age and is at the evolutionary stage for a significant fraction of CRs to escape, we aim to understand $\gamma$-ray emission in the...
  5. [0.8155] Equations of state in the Hartle-Thorne model of neutron stars selecting accepta (чанк #1)
     Strong restrictions of the neutron star parameters M (mass) and a (spin) arise due to fitting the frequency pairs admitted by the RS model to the observed data ...
```
</details>

**Яка стратегія дає осмисленіші чанки?** Semantic: кожен чанк складається з цілих речень. Fixed обриває текст на *"Their goal"* і починає наступний чанк з середини речення.

**Розрізані речення.** У fixed 209 з 243 чанків закінчуються посеред речення, у semantic 0. Думка ділиться між двома векторами, і жоден її повністю не описує, тому збіг з запитом слабшає (на запиті про нейтронні зорі 0.863 у fixed проти 0.867 у semantic).

**Overlap.** Чим він більший, тим більше чанків: 212, 243, 305, 436 при overlap 0, 10, 20, 30. Тексту в чанках в 1.00, 1.23, 1.59 і 2.30 раза більше за оригінал. Межі покриті краще, але векторів і дублікатів у видачі більше.

---

## Частина 5. Гібридний пошук (додатково)

BM25 (`rank_bm25`) по `title + abstract + authors`, вектор через Pinecone (топ-10), об'єднання через RRF: $score(d) = \sum \frac{1}{k + rank}$, $k = 60$.

<details><summary>Вивід 06_hybrid_search.py</summary>

```
BM25-індекс побудовано: 10000 документів


Запит (точний термін): 'BERT fine-tuning'

BM25 топ-5:
  1. It's not Greek to mBERT: Inducing Word-Level Translations from Multilingual | cs.CL | 2020
  2. Robust Transfer Learning with Pretrained Language Models through Adapters | cs.CL | 2021
  3. Annotated Job Ads with Named Entity Recognition | cs.CL | 2023
  4. A Curious Case of Searching for the Correlation between Training Data and A | cs.LG | 2024
  5. Gender-tuning: Empowering Fine-tuning for Debiasing Pre-trained Language Mo | cs.CL | 2023

Vector топ-5:
  1. A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning an | cs.AI | 2025
  2. Beyond Fixed Morphologies: Learning Graph Policies with Trust Region Compen | cs.LG | 2025
  3. PolySwarm: A Multi-Agent Large Language Model Framework for Prediction Mark | cs.AI | 2026
  4. Self-adapting Robotic Agents through Online Continual Reinforcement Learnin | cs.RO | 2026
  5. Robotic Table Tennis with Model-Free Reinforcement Learning | cs.LG | 2020

Hybrid RRF (k=60) топ-5:
  1. [0.0164] It's not Greek to mBERT: Inducing Word-Level Translations from Multilingual | cs.CL | 2020  <- BM25 #1
  2. [0.0164] A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning an | cs.AI | 2025  <- Vec #1
  3. [0.0161] Robust Transfer Learning with Pretrained Language Models through Adapters | cs.CL | 2021  <- BM25 #2
  4. [0.0161] Beyond Fixed Morphologies: Learning Graph Policies with Trust Region Compen | cs.LG | 2025  <- Vec #2
  5. [0.0159] Annotated Job Ads with Named Entity Recognition | cs.CL | 2023  <- BM25 #3

  у гібридному топ-5: знайдено обома методами: 0, відсутні в топ-5 обох методів: 0

  RRF k=1 топ-5:
  1. [0.5000] It's not Greek to mBERT: Inducing Word-Level Translations from Multilingual | cs.CL | 2020  <- BM25 #1
  2. [0.5000] A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning an | cs.AI | 2025  <- Vec #1
  3. [0.3333] Robust Transfer Learning with Pretrained Language Models through Adapters | cs.CL | 2021  <- BM25 #2
  4. [0.3333] Beyond Fixed Morphologies: Learning Graph Policies with Trust Region Compen | cs.LG | 2025  <- Vec #2
  5. [0.2500] Annotated Job Ads with Named Entity Recognition | cs.CL | 2023  <- BM25 #3

  топ-5 при k=1 і k=60 однаковий: True


Запит (ім'я автора): 'Yann LeCun convolutional networks'

BM25 топ-5:
  1. Very Deep Convolutional Networks for Text Classification | cs.CL | 2016
  2. Prediction Under Uncertainty with Error-Encoding Networks | cs.AI | 2017
  3. Compact and Optimal Deep Learning with Recurrent Parameter Generators | cs.CV | 2021
  4. Revisiting Feature Prediction for Learning Visual Representations from Vide | cs.CV | 2024
  5. Deep Learning for Metagenomic Data: using 2D Embeddings and Convolutional N | cs.CV | 2017

Vector топ-5:
  1. Feed Forward and Backward Run in Deep Convolution Neural Network | cs.CV | 2017
  2. KAConvNet: Kolmogorov-Arnold Convolutional Networks for Vision Recognition | cs.CV | 2026
  3. Learning Compact Convolutional Neural Networks with Nested Dropout | cs.CV | 2014
  4. Unifying local and non-local signal processing with graph CNNs | cs.CV | 2017
  5. Deep Unitary Convolutional Neural Networks | cs.LG | 2021

Hybrid RRF (k=60) топ-5:
  1. [0.0304] KAConvNet: Kolmogorov-Arnold Convolutional Networks for Vision Recognition | cs.CV | 2026  <- BM25 #10 + Vec #2
  2. [0.0303] Deep Unitary Convolutional Neural Networks | cs.LG | 2021  <- BM25 #7 + Vec #5
  3. [0.0164] Very Deep Convolutional Networks for Text Classification | cs.CL | 2016  <- BM25 #1
  4. [0.0164] Feed Forward and Backward Run in Deep Convolution Neural Network | cs.CV | 2017  <- Vec #1
  5. [0.0161] Prediction Under Uncertainty with Error-Encoding Networks | cs.AI | 2017  <- BM25 #2

  у гібридному топ-5: знайдено обома методами: 2, відсутні в топ-5 обох методів: 0

  RRF k=1 топ-5:
  1. [0.5000] Very Deep Convolutional Networks for Text Classification | cs.CL | 2016  <- BM25 #1
  2. [0.5000] Feed Forward and Backward Run in Deep Convolution Neural Network | cs.CV | 2017  <- Vec #1
  3. [0.4242] KAConvNet: Kolmogorov-Arnold Convolutional Networks for Vision Recognition | cs.CV | 2026  <- BM25 #10 + Vec #2
  4. [0.3333] Prediction Under Uncertainty with Error-Encoding Networks | cs.AI | 2017  <- BM25 #2
  5. [0.2917] Deep Unitary Convolutional Neural Networks | cs.LG | 2021  <- BM25 #7 + Vec #5

  топ-5 при k=1 і k=60 однаковий: False


Запит (перефразування): 'making computers understand human emotions from text'

BM25 топ-5:
  1. Fair and Unbiased Algorithmic Decision Making: Current State and Future Cha | cs.LG | 2019
  2. EmoGenie: Benchmarking and Evaluating Expressed and Evoked Emotion in T2I G | cs.CV | 2026
  3. Do Text Simplification Systems Preserve Meaning? A Human Evaluation via Rea | cs.CL | 2023
  4. Evaluation of Generative Models for Emotional 3D Animation Generation in VR | cs.HC | 2025
  5. Performance Evaluation of Supervised Machine Learning Techniques for Effici | cs.IR | 2019

Vector топ-5:
  1. Performance Evaluation of Supervised Machine Learning Techniques for Effici | cs.IR | 2019
  2. "We care": Improving Code Mixed Speech Emotion Recognition in Customer-Care | cs.AI | 2023
  3. Emotion Recognition and Generation: A Comprehensive Review of Face, Speech, | cs.LG | 2025
  4. Development of computational models for emotional diary text analysis to su | cs.CY | 2017
  5. EmpathBERT: A BERT-based Framework for Demographic-aware Empathy Prediction | cs.LG | 2021

Hybrid RRF (k=60) топ-5:
  1. [0.0318] Performance Evaluation of Supervised Machine Learning Techniques for Effici | cs.IR | 2019  <- BM25 #5 + Vec #1
  2. [0.0304] EmoGenie: Benchmarking and Evaluating Expressed and Evoked Emotion in T2I G | cs.CV | 2026  <- BM25 #2 + Vec #10
  3. [0.0164] Fair and Unbiased Algorithmic Decision Making: Current State and Future Cha | cs.LG | 2019  <- BM25 #1
  4. [0.0161] "We care": Improving Code Mixed Speech Emotion Recognition in Customer-Care | cs.AI | 2023  <- Vec #2
  5. [0.0159] Do Text Simplification Systems Preserve Meaning? A Human Evaluation via Rea | cs.CL | 2023  <- BM25 #3

  у гібридному топ-5: знайдено обома методами: 2, відсутні в топ-5 обох методів: 0

  RRF k=1 топ-5:
  1. [0.6667] Performance Evaluation of Supervised Machine Learning Techniques for Effici | cs.IR | 2019  <- BM25 #5 + Vec #1
  2. [0.5000] Fair and Unbiased Algorithmic Decision Making: Current State and Future Cha | cs.LG | 2019  <- BM25 #1
  3. [0.4242] EmoGenie: Benchmarking and Evaluating Expressed and Evoked Emotion in T2I G | cs.CV | 2026  <- BM25 #2 + Vec #10
  4. [0.3333] "We care": Improving Code Mixed Speech Emotion Recognition in Customer-Care | cs.AI | 2023  <- Vec #2
  5. [0.2500] Do Text Simplification Systems Preserve Meaning? A Human Evaluation via Rea | cs.CL | 2023  <- BM25 #3

  топ-5 при k=1 і k=60 однаковий: False
```
</details>

| Запит | BM25 | Vector | Hybrid (RRF) |
|---|---|---|---|
| BERT fine-tuning | 5 з 5 про BERT | 0 з 5, видав RL і роботів | 3 з 5 від BM25 |
| Yann LeCun convolutional networks | 2 статті LeCun | 5 з 5 про CNN, LeCun немає | CNN-статті, 1 стаття LeCun |
| making computers understand human emotions from text | збіг за "making" і "human" | 5 з 5 про емоції | 3 з 5 про емоції |

**Який метод кращий?** Залежить від запиту. На точному терміні і на авторі виграв BM25, бо рідкісне слово чи прізвище має бути в тексті, а авторів у векторі взагалі немає. На перефразуванні виграв вектор. Гібрид ні разу не був найгіршим, але і кращим за найсильніший метод не став: RRF не знає, який список хороший, і бере з обох.

**Чи є в гібриді документи, яких нема в топ-5 окремих методів?** За місцями так: `KAConvNet` (#10 у BM25, #2 у вектора) в гібриді перший. Документ з обох списків набирає ~0.030, а лідер тільки одного списку ~0.016.

**k=60 і k=1.** При k=60 вирішує згода методів, при k=1 вирішують перші місця кожного методу, і `KAConvNet` падає на третє місце.

---

## Частина 6. Аналіз і висновки

### 1. Семантичний пошук і пошук за ключовими словами

Семантичний пошук добре працює, коли людина описує ідею своїми словами: "teaching machines to recognize objects in pictures" знайшов розпізнавання об'єктів, а запит про емоції знайшов emotion recognition, хоча ключові слова ніде не збігаються. BM25 кращий для точних речей: назв моделей і абревіатур (BERT), імен авторів (LeCun), номерів і формул. Ембеддинг такі речі або розмиває до загальної теми, або взагалі не містить. Правило таке: шукаєш за змістом, бери вектор; шукаєш конкретне слово чи ім'я, бери BM25; не знаєш заздалегідь, бери гібрид.

### 2. Вплив розміру чанка

Чанк на 10-15 слів це пів речення без контексту: ембеддинг нестабільний, векторів стає дуже багато, а знайдений шматок занадто короткий, щоб з нього була користь. Чанк на 500+ слів не влазить у ліміт моделі (512 токенів), і хвіст мовчки відкидається, у нашій вибірці так втрачено кінець у 52 анотацій. До того ж великий вектор усереднює кілька тем, і точне питання програє загальній схожості. Оптимум залежить від моделі і задачі: для пошуку статей за темою підходить вся анотація, для пошуку фактів і RAG краще 100-300 токенів по межах речень з невеликим overlap.

### 3. L2-індекс для нормалізованих векторів

Для $\|a\| = \|b\| = 1$:

$$\|a - b\|^2 = \|a\|^2 - 2\,a\cdot b + \|b\|^2 = 2 - 2\cos(a, b)$$

Відстань $\sqrt{2 - 2\cos}$ тим менша, чим більший косинус, тому euclidean-індекс відсортує результати так само, як cosine. Це підтверджує і мій експеримент, топ-5 однакові. Зміниться тільки шкала: замість 0.86 (більше краще) буде відстань ~0.28 (менше краще). Зламатись може хіба що код з порогами на score. Справжня проблема була б з ненормалізованими векторами.

### 4. Обмеження Pinecone Starter і 10 млн статей

На безкоштовному тарифі тільки serverless в `aws us-east-1`, обмежена кількість індексів (мені знадобилось три), 40 KB метаданих на вектор (тому анотація обрізана до 500 символів), обмежений розмір запиту (тому батчі) і обмежене сховище. Для 10 млн статей самі вектори займуть ~31 GB, це вже платний тариф або квантизація. Ембеддинги на CPU рахувались би ~300 годин (у мене 10 000 за 18 хв), тому потрібен GPU і збереження проміжних результатів. Тексти краще тримати окремо (S3, Postgres), а в Pinecone тільки id і поля для фільтрів, завантажувати паралельно або через bulk import. BM25 замінити на Elasticsearch або sparse-вектори в Pinecone, бо `rank_bm25` тримає все в пам'яті.
