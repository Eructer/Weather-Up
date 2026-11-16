# Weather-Up

Machine Learning Model trained on weather data to predict how you should dress for the weather

# Table of contents
- [Weather-Up](#weather-up)
  - [Motivation](#motivation)
  - [Goal](#goal)
  - [Proposed Method](#proposed-method)
    - [Preface](#preface)
    - [Model Architecture](#model-architecture)
  - [Evaluation](#evaluation)
- [Expected Contributions](#expected-contributions)
- [Authors](#authors)

## Motivation

Trying to decide what to wear on any day can be the hardest decision to make every day. What if there could be a guide on what to wear? Help you make up your mind quicker? Thats what this project is about. With historic weather data and some modified data, we can use Machine Learning to predict what to wear given the weather.

## Goal

Our goal is to have a Machine Learning Model that can enable people to make more efficient clothing decisions based on the weather. Including with this we need the model make accurate predictions to ensure you are making the right decisions.

## Proposed Method

### Preface

Since there are multiple things you can wear (shoes, socks, shirts, hoodies, hats, etc), we need to be able to account for combinations of things. Assuming that people have common sense, we will assume that people have the basics already picked out.

- Socks
- Shoes
- Undergarments

The Model will handle the rest,

- Shirt
- Long-T
- Tank top
- Shorts
- Pants
- Hoodie
- Hat
- Beenie

Sometimes weather needs a combination of clothes (Maybe it rains half the day, you don't need a thick hoodie all day)

### Model Architecture

To handle the combinations of clothes we can use a **Multi-Layer-Perceptron (MLP)** with **Softmax** activation functions. (In `scikit-learn` this is a **MLPClassifier**)

The key attribute here that can allow use to get combinations is the **Softmax** activation function. **Softmax** allows for turning inputs (logits) into a probability distribution. We can set a threshold on what level of probability counts towards the final prediction.

Using supervised learning of course. (Our data is labeled)

## Evaluation

To properly evaluate the model, we can use the idea of train/validation/test split ratios on our data. This gives us ample sizes of data for training, validating, and testing. After a model has been trained, we can give it new fresh data and compare the prediction with what we would expect to wear. This type of evaluation is also known as **Human-in-the-loop** evaluation.

# Expected Contributions

Everyone working on this project is expected to work on every bit, including the following

- Machine Learning Model
- Data collection
- Data cleanining
- Data modification
- Testing/Validation
- Endpoint for testing

# Authors

- Eructer/Jay M
- Isaac
