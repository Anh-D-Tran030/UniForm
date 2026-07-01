@echo off
cd /d A:\
"C:\Users\thanh\anaconda3\envs\LayoutLM\python.exe" "A:\RealForm\scripts\benchmark_colflor_document_similarity.py" --dataset test --encode-batch-size 1 --query-chunk-size 4 --gallery-chunk-size 128 --score-batch-size 16 --log-file "A:\RealForm\processed\colflor_document_similarity_benchmark\test\colflor_benchmark_full.log" > "A:\RealForm\logs\colflor-test.out.log" 2> "A:\RealForm\logs\colflor-test.err.log"
