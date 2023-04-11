# COVID Chatbot Application
A deep learning based chatbot for COVID-19

The chatbot is available at: [https://huggingface.co/spaces/waleedgeo/covidbot](https://huggingface.co/spaces/waleedgeo/covidbot).

[![Chatbot](chatbot_pic.png)](https://huggingface.co/spaces/waleedgeo/covidbot)

The app was created using [Gradio](https://gradio.app/) and [Hugging Face](https://huggingface.co/spaces).


# Data sources

The data for the chatbot was collected from the following sources:

1. https://www.gov.uk/guidance/people-with-symptoms-of-a-respiratory-infection-including-covid-19
2. https://www.nhs.uk/conditions/covid-19/covid-19-symptoms-and-what-to-do/
3. https://www.fda.gov/emergency-preparedness-and-response/coronavirus-disease-2019-covid-19/covid-19-frequently-asked-questions
4. https://www.coronavirus.gov.hk/
5. https://www.who.int/europe/news-room/fact-sheets/item/post-covid-19-condition
6. https://www.who.int/southeastasia/outbreaks-and-emergencies/covid-19/questions/post-covid-19-q-a
7. https://www.nhs.uk/conditions/covid-19/long-term-effects-of-covid-19-long-covid/
8. https://www.transperfect.com/dataforce/covid-19-chatbot
9. https://www.ha.org.hk/visitor/ha_visitor_index.asp?Content_ID=254422&Lang=ENG&Dimension=100

# LSTM Model Summary

The model was trained using LSTM model for 200 epochs. The model was trained using the Adam optimizer with a learning rate of 0.001. The model was trained on a GPU (NVEDIA RTX 3070, Compute Unit 8.6). The summary of model is:

![Model](model_summary.png)

# Accuracy

The model is trained on the [COVID-19 Open Research Dataset](https://pages.semanticscholar.org/coronavirus-research) and achieves an accuracy of 0.99.

![Accuracy](model_accuracy.png)

---