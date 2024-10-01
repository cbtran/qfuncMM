FROM rocker/verse:4.3

### install python
# copy from jupyter stacks
# roughly around the same time as rocker/verse:4.2
COPY --from=jupyter/base-notebook:x86_64-python-3.10 /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

### install quarto
# versions: https://quarto.org/docs/download/_download.json
RUN curl --silent -L --fail \
        https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.57/quarto-1.5.57-linux-amd64.deb > /tmp/quarto.deb && \
    apt update && \
    apt install -y --no-install-recommends /tmp/quarto.deb tmux gdb r-base-dev clangd

USER rstudio

RUN pip install radian && \
    install2.r languageserver httpgd && \
    installGithub.r ManuelHentschel/vscDebugger


RUN install2.r ggpubr corrplot latex2exp Matrix xtable ggridges kableExtra RcppEnsmallen RcppArmadillo tictoc

USER root