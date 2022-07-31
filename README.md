# Kaggle

Some of my code snippets of my data science course (kaggle competitions)

## Set up environment

```bash
git clone --recurse-submodules https://gitlab.com/wc-yat/kaggle.git
cd kaggle
python -m venv venv
source venv/bin/activate
pip install pandas ./learntools
sed -i '/learntools/d' requirements.txt
pip install -r requirements.txt
```
