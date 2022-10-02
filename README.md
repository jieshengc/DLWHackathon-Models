# DLWHackathon-Models

By team chickenramen.

For web application implementation, refer to https://github.com/szekee/chickenramen

## Problem Statement

As Singapore moves towards Smart Nation initiative, bad actors continue to use phishing attacks to obtain precious information from unsuspecting users. Furthermore, there is a exponential increase of phishing attacks due to Covid-19 and the use of links, QR codes and more. Our project aims to use ML models to detect and prevent such attacks.

- 25% of Singapore’s population is predicted to be aged 65 and older by 2030, versus 14.4% in 2019. 
- As Singapore marches towards a Smart Nation Initiative. It has to take into account the massive eldery population that are vulnerable to such phishing attacks.
- Besides the elderly, COVID-19 taught our nation being to trust QR codes and URL redirects as we relied on these technologies to facilitate safe-entry access. 
- As a result of these technological adoption, there was been a 94% increase in scam frequency within the first half of 2022 alone as compared to 2021.
- However, the reliance on the aforementioned technology has also increased the amount of phishing attacks brought about by bad actors in society.
- Furthermore, the use of QR code to register for links have given rise to “Quishing”. It occurs when an individual uses QR code to trick people to share personal or financial information. 

## Model Building
Our team complied a (non-exhaustive) list of features and built feature extraction pipelines to ingest URLs to be broken down into components for our machine learning models:

- URL domain
- Presence of IP address
- Presence of ‘@’ symbol
- URL length
- URL depth
- Redirection
- Existence of HTTPS in domain
- Presence of shortened URLs
- Presence of ‘-’ symbol
- DNS record
- Domain age
- End period of domain
- IFrame redirection
- Status bar customization
- Status of right click
- Number of website forwardings

Our team also made use of multiple models to build 5 main models each with Different Features and Feature Extraction methods.
Our final model is an ensemble comprising of the 5 individual Models:

| Individual Models  | Test Accuracy |
| ------------- | ------------- |
| Model 1 (JS)  | 94.6%  |
| Model 2 (JL)  | 87.0%  |
| Model 3 (SK)  | 99.4%  |
| Model 4 (HX)  | 86.4%  |
| Model 5 (WD)  | 91.4%  |

## Datasets
Datasets were obtained from various sources.
https://www.kaggle.com/code/hamzamanssor/detection-malicious-url-using-ml-models/data 
https://research.aalto.fi/en/datasets/phishstorm-phishing-legitimate-url-dataset 
https://www.kaggle.com/code/mpwolke/phishing-detection-with-same-bait 
https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/tree/master/DataFiles 

All datasets comprises of an URL as an input variable and a label as an output variable → Features of the URL will be extracted by the different models → Features will be used in the determination of whether URL is legitimate or phishing
(Dataset summary of the Models is shown below - train-test split is a variation of 80-20 and 70-30)

| Individual Models  | Data |
| ------------- | ------------- |
| Model 1 (JS)  | 48,009 legitimate vs 48,009 phishing |
| Model 2 (JL)  | 428,103 benign vs 94,111 phishing |
| Model 3 (SK)  | 35,378 benign vs 35,378 phishing |
| Model 4 (HX)  | 35,378 benign vs 14,859 phishing |
| Model 5 (WD)  | 5,714 legitimate vs 5,714 phishing |

Some general functions used to extract the features from the URLs include, but are not limited to:
Urlparse, tld, tldextract, re, ipaddress, sys, whois, datetime 




