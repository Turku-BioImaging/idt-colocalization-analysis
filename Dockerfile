FROM continuumio/miniconda3:4.10.3-alpine

LABEL org.opencontainers.image.authors="Junel Solis, Image Data Team, Turku BioImaging"
LABEL org.opencontainers.image.url="https://github.com/turku-bioimaging/idt-colocalization-analysis"
LABEL org.opencontainers.image.source="https://github.com/turku-bioimaging/idt-colocalization-analysis"
LABEL org.opencontainers.image.title="Image Colocalization Analysis"

WORKDIR /code

RUN addgroup -S pythonuser && adduser -S pythonuser -G pythonuser \
    && chown -R pythonuser:pythonuser /opt/conda 

COPY environment.yml analyze.py functions.py /code/
RUN chmod o=rx environment.yml analyze.py functions.py

USER pythonuser

RUN conda env create -f environment.yml && conda clean --all

ENV PATH /opt/conda/envs/idt-colocalization-analysis/bin:$PATH
RUN /bin/bash -c "source activate idt-colocalization-analysis"

CMD [ "python", "analyze.py" ]