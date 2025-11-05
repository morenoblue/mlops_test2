FROM python:3.11-slim

# note(@morenoblue): This variable is to stop Python from writing 
#                    __pycache__/*.pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 

# note(@morenoblue): This makes stdout/stderr unbuffered → logs appear immediately.
ENV PYTHONUNBUFFERED=1        

# note(@morenoblue): The variables below help to prevent CPU subscriptions due 
#                    to how libraries like numpy, scipy and pandas work
ENV OMP_NUM_THREADS=1         
ENV MKL_NUM_THREADS=1         
ENV NUMEXPR_NUM_THREADS=1     

WORKDIR /app
COPY requirements.txt README.md /app/
COPY src/ /app/src/

# note(@morenoblue): The --no-cache-dir flag is just to tell pip to avoid caches
#                    since we would like to save our sweet sweet space for other stuff.
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "/app/src/train.py"]

