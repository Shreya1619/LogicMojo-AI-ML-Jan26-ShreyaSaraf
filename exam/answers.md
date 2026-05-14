# AI/ML Deep Learning Exam

Duration: 90 minutes  
Total marks: 100

## Instructions

- Attempt all sections.
- For coding, edit only `coding/student.py`.
- Fixed datasets are in `coding/data/`.
- Fixed sample inputs and outputs are in `coding/sample_io.md`.
- Your code should be vectorized where reasonable and should avoid data leakage.
- PyTorch code must run on CPU. Do not assume a GPU is available.
- Write clear, concise answers for descriptive questions.

## Section A: MCQ

Each question carries 1 mark.

1. You are building customer features for a purchase classifier. Which operation most clearly causes target leakage?
   - A. One-hot encoding `event_type`
   - B. Using the target column itself as an input feature
   - C. Filling missing `price` values with the median price
   - D. Counting purchases per `customer_id`

   
   Answer: B
   Reason: Because it introduces information into the training data that will not be available at the time of prediction, leading to  high accuracy during training and low testing accuracy.



2. In pandas, why is `df.loc[mask].copy()` often safer before adding engineered columns?
   - A. It always makes code faster
   - B. It avoids mutating a view unexpectedly and reduces chained-assignment bugs
   - C. It automatically removes duplicate rows
   - D. It changes all object columns to categorical

   Answer: B
   Reason: Calling .copy() gives you a new dataframe and we can avoid making unsaved changes. 


3. A CNN receives input shape `(batch, 3, 64, 64)`. A `Conv2d(3, 16, kernel_size=3, stride=2, padding=1)` produces which spatial shape?
   - A. `(16, 31, 31)`
   - B. `(16, 32, 32)`
   - C. `(16, 33, 33)`
   - D. `(3, 32, 32)`

   Answer: B
   Reason: floor((64 + 2 - 3) / 2) + 1 = floor(63/2) + 1 = 31 + 1 = 32
Channels become 16. 

4. Why should `model.train()` be called before a PyTorch training loop?
   - A. It freezes all gradients
   - B. It switches layers like Dropout and BatchNorm into training behavior
   - C. It sends the model to GPU
   - D. It resets all weights

    Answer: B
    Reason:  In PyTorch is essential to explicitly set the model to training mode, activating specific behaviors for layers like Dropout and BatchNorm that behave differently during training vs. evaluation

5. `nn.CrossEntropyLoss` in PyTorch expects:
   - A. probabilities after softmax and one-hot labels
   - B. raw logits and integer class labels
   - C. sigmoid outputs and float labels only
   - D. normalized embeddings and cosine labels

  


6. A binary classifier has 98% accuracy on a dataset where 98% of labels are negative. What is the best immediate concern?
   - A. The model is definitely excellent
   - B. Accuracy may hide poor recall on the positive class
   - C. The learning rate must be too high
   - D. BatchNorm is impossible to use

   Answer: B


7. If class counts are `[900, 100]`, why might inverse-frequency sampling help?
   - A. It removes the minority class
   - B. It makes every mini-batch contain only minority samples
   - C. It increases the chance of seeing minority-class samples during training
   - D. It changes the target labels into probabilities

   Answer:B
   Reason:A model that just predicts "negative" for everything gets 98% accuracy without learning anything.Accuracy looks great but recall on the positive class is 0%. 

8. In a CNN, increasing receptive field usually helps because:
   - A. the model can use larger context from the image
   - B. it removes the need for nonlinearities
   - C. it guarantees no overfitting
   - D. it makes all filters identical

   Answer:A
   Reason:A larger receptive field enables the model to see more of the surrounding environment, which is vital for accurately interpreting objects based on their contex


9. Which augmentation is usually unsafe for digit classification if labels must remain exact?
   - A. Small random translation
   - B. Mild brightness jitter
   - C. Horizontal flip for asymmetric digits
   - D. Small random rotation

   Answer:C
   Reason:"6" vs "9", or "2" vs "5"

10. Why is `optimizer.zero_grad()` normally called each iteration?
    - A. PyTorch accumulates gradients by default
    - B. It resets the model weights to zero
    - C. It deletes the loss function
    - D. It disables backpropagation

    Answer:A 

11. Which metric is most useful when false negatives are very costly?
    - A. Recall
    - B. Training loss only
    - C. Number of model parameters
    - D. Number of convolution filters only

    Answer:A


12. In transfer learning with a pretrained CNN, a common first approach is:
    - A. discard all pretrained weights
    - B. freeze early feature layers and train a new classifier head
    - C. train only on labels from ImageNet
    - D. remove all convolution layers

    Answer: B


13. Batch normalization uses different statistics in training and evaluation. What should be done before validation?
    - A. `model.eval()`
    - B. `loss.backward()`
    - C. `optimizer.step()`
    - D. `model.zero_parameters()`

    Answer:A
    

14. In `df.groupby("customer_id").agg(unique_products=("product_id", "nunique"))`, what does `nunique` compute?
    - A. the number of distinct non-null products per customer
    - B. the total number of rows per customer
    - C. the most frequent product per customer
    - D. the average product price per customer

    Answer:A
    Reason: nunique = number of unique non-null values. So it counts how many distinct products each customer interacted with.



15. What does padding in convolution mainly control?
    - A. number of classes
    - B. spatial size and edge information handling
    - C. optimizer type
    - D. random seed

    Answer:B


16. If training loss decreases but validation loss increases for many epochs, the likely issue is:
    - A. underfitting
    - B. overfitting
    - C. missing `import torch`
    - D. too few labels in the output layer only

    Ans: B


17. What is the main benefit of using `DataLoader` with mini-batches?
    - A. It converts classification to regression
    - B. It handles batching, shuffling, and efficient iteration
    - C. It removes the need for labels
    - D. It guarantees perfect generalization

    Ans: B

18. Which statement about logits is correct?
    - A. Logits are raw unnormalized scores before softmax
    - B. Logits must sum to 1
    - C. Logits are always binary
    - D. Logits are labels after encoding

    Ans: A

19. In image classification, global average pooling can help by:
    - A. increasing spatial dimensions
    - B. reducing parameters compared with a large flatten + linear head
    - C. removing all channels
    - D. preventing gradient computation

    Ans:B

20. In pandas, `groupby().agg()` is preferred over Python loops mainly because:
    - A. it is usually more concise and faster for tabular aggregation
    - B. it prevents all data leakage automatically
    - C. it trains neural networks
    - D. it always uses GPU acceleration

    Ans:A

## Section B: Conceptual Questions

Answer any 4 questions. Each question carries 5 marks.

1. You are given product interaction logs with `customer_id`, `event_type`, `product_id`, `price`, `quantity`, and `is_returned`. Describe a pandas feature engineering pipeline that creates one row per customer using counts, revenue, return behavior, unique products, and conversion-rate style features.

**Ans:**

Suppose following is the raw data:


| customer_id | event_type | product_id | price | quantity | is_returned |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 101 | view | A1 | 20 | 1 | 0 |
| 101 | purchase | A1 | 20 | 1 | 0 |
| 101 | purchase | B2 | 50 | 1 | 1 |
| 102 | view | C3 | 15 | 1 | 0 |
| 102 | view | D4 | 30 | 1 | 0 |
| 103 | purchase | E5 | 100 | 1 | 0 |


**The Approach:** We use `groupby('customer_id')` to put all rows for a specific customer into a "bucket." Then, we apply specific mathematical rules to the columns inside that bucket:
    *   **Sum:** Add up all the `revenue` and `is_returned` values to get totals.
    *   **Conditional Count:** We look at the `event_type` column. We count how many times "purchase" appears and how many times "view" appears.
    *   **Unique Count:** We look at the `product_id` column and count the number of *distinct* items interacted with (`nunique`).
    *   **return_rate** = Total Returns / Total Purchases
    *   **conversion_rate** = Total Purchases / Total Views
*   **Handling Errors:** Apply `.fillna(0)` at the end to ensure that if a user has zero purchases, dividing by zero doesn't break our dataset.

##### Steps:
df['revenue'] = df['price'] * df['quantity']
summary = df.groupby('customer_id').agg(
    total_spent=('revenue', 'sum'),
    purchases=('event_type', lambda x: (x == 'purchase').sum()),
    views=('event_type', lambda x: (x == 'view').sum()),
    returns=('is_returned', 'sum'),
    unique_products=('product_id', 'nunique')
)

summary['return_rate'] = (summary['returns'] / summary['purchases']).fillna(0)
summary['conv_rate'] = (summary['purchases'] / summary['views']).fillna(0)

| customer_id | total_spent | purchases | views | returns | unique_products | return_rate | conv_rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **101** | 90 | 2 | 1 | 1 | 2 | 0.5 | 2.0 |
| **102** | 0 | 0 | 2 | 0 | 2 | 0.0 | 0.0 |
| **103** | 100 | 1 | 0 | 0 | 1 | 0.0 | inf |


---

2. A CNN trained on 28x28 grayscale images reaches 99% training accuracy but only 82% validation accuracy. Explain at least four likely causes or fixes.

    **Ans:**
        The large gap (99% train vs 82% val) is a classic overfitting signature. Likely causes and fixes:
        i. Model too large for the data — too many parameters relative to training examples. Fix: reduce model depth/width or use a simpler architecture.
        ii. No regularization — without Dropout or weight decay, nothing penalizes the model for memorizing. Fix: add nn.Dropout, add weight_decay to the optimizer.
        iii. Training too long without early stopping — validation loss starts rising while training loss keeps falling. Fix: monitor validation loss and stop when it stops improving.
        iv. Insufficient data diversity — model hasn't seen enough variation. Fix: apply data augmentation (random rotations, translations, flips where label-safe) to artificially increase variety.
        v. Not calling model.eval() during validation — Dropout stays active, giving inconsistent results. Fix: always wrap validation in model.eval() and torch.no_grad().

---

3. Explain the difference between logits, probabilities, and predicted class labels in a multi-class PyTorch classifier. Include where `CrossEntropyLoss` fits in.

**Ans:**
* Logits are the raw, unconstrained outputs of the final linear layer — real numbers with no bounds (e.g. [-1.2, 3.4, 0.7]).
* Probabilities are obtained by applying softmax to logits. The output values are all positive and sum to 1, representing the model's confidence per class (e.g. [0.05, 0.88, 0.07]).
* Class labels are the final prediction — the integer index of the highest probability, obtained via argmax (e.g. 1).
* nn.CrossEntropyLoss expects raw logits (not softmax outputs) because it applies LogSoftmax internally in a numerically stable way. Passing softmax outputs would apply it twice and produce incorrect gradients. During inference, to get probabilities you apply torch.softmax(logits, dim=1), and for the predicted class, torch.argmax(logits, dim=1).

---

4. A medical image dataset has 5,000 normal samples and 300 disease-positive samples. Propose a training and evaluation strategy that handles imbalance responsibly.

**Ans:**
With a 5000:300 imbalance (about 17:1 ratio), naive training will create a model biased toward the majority class. A responsible strategy includes:

Training:

- Use inverse-frequency sampling weights (WeightedRandomSampler in PyTorch) so each mini-batch reflects roughly equal class representation.
- Alternatively, use class-weighted loss: pass the weight parameter to CrossEntropyLoss to penalize disease-positive errors more heavily.
- Apply augmentation specifically to minority class samples to boost effective diversity.

Evaluation:

- Never use accuracy alone; a model that predicts "normal" will always achieve 94%.
- Use recall (sensitivity) as the main metric; missing a disease case is far worse than a false alarm.
- Report precision, recall, F1 for each class, and the complete confusion matrix.
- Use a held-out stratified test set that maintains the real-world class ratio.

The goal is to minimize false negatives (missed disease cases), even if it means accepting some false positives.

---

5. You are fine-tuning a pretrained CNN on a small custom dataset. Explain when you would freeze layers, when you would unfreeze layers, and how you would choose learning rates.


---

## Section C: Coding

Complete `coding/student.py`. tests are in `coding/tests/test_student.py`.
Use `coding/sample_io.md` for fixed examples of input and output.

### Task 1: Pandas Feature Engineering

Implement:

```python
build_customer_product_features(events)
```

Requirements:

- Build one row per `customer_id`.
- Create these columns:
  - `customer_id`
  - `view_count`
  - `cart_count`
  - `purchase_count`
  - `non_returned_purchase_count`
  - `gross_revenue`
  - `return_rate`
  - `unique_products`
  - `avg_order_value`
  - `cart_to_purchase_rate`
  - `view_to_purchase_rate`
- Treat revenue as `price * quantity` for non-returned purchase rows.
- If a denominator is 0 for a rate feature, return 0 for that rate.

Marks: 18

### Task 2: Product Revenue Ranking

Implement:

```python
top_products_by_revenue(events, top_n=3)
```

Requirements:

- Use only non-returned purchase rows for `gross_revenue`.
- `gross_revenue = price * quantity`.
- Return these columns:
  - `product_id`
  - `non_returned_units`
  - `gross_revenue`
  - `unique_buyers`
  - `return_rate`
- Sort by `gross_revenue` descending, then `product_id` ascending.
- Return only the top `top_n` rows.
- Use 0 for `return_rate` when there are no purchases.

Marks: 12

### Task 3: Imbalance and CNN Shape Utilities

Implement:

```python
make_balanced_sampler_weights(labels)
conv2d_output_shape(input_hw, kernel_size, stride=1, padding=0, dilation=1)
```

Marks: 10

### Task 4: PyTorch CNN

Implement `TinyCnn` and `count_trainable_parameters`.

Expected architecture:

- Input: `(batch, 1, 28, 28)`
- Conv block 1: Conv2d 1 to 8, kernel 3, padding 1; BatchNorm; ReLU; MaxPool2d 2
- Conv block 2: Conv2d 8 to 16, kernel 3, padding 1; BatchNorm; ReLU; MaxPool2d 2
- Classifier: Dropout 0.2; Linear from `16 * 7 * 7` to `num_classes`
- For `TinyCnn(num_classes=5)`, `count_trainable_parameters(model)` should return `5221`.

Marks: 10

### Task 5: PyTorch Training Loop and Evaluation Metrics

Implement:

```python
train_one_epoch(model, dataloader, optimizer, criterion, device)
confusion_matrix(y_true, y_pred, num_classes)
macro_f1_from_confusion(cm)
```

`train_one_epoch` should return:

```python
{"loss": average_loss, "accuracy": average_accuracy}
```

Marks: 10
