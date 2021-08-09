ARG base_image
FROM $base_image

ARG config_path

LABEL maintainer="ink@myrabr.com"

# ADD lib/wheels wheels
# RUN pip install ./wheels/*

ADD dist wheels
RUN pip install ./wheels/*

ADD requirements.txt .
RUN pip install --quiet -r requirements.txt

ADD $config_path/spark-defaults.conf $SPARK_HOME/conf/
