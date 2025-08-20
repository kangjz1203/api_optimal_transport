python scripts/extract_code_context.py \
    --input_file "code_editing_old_to_new.json" \
    --output_file "old_to_new.json" \
    --corpus_dir "/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/Corpus/library_source_code/version_corpus"

python scripts/extract_code_context.py \
    --input_file "code_editing_new_to_old.json" \
    --output_file "new_to_old.json" \
    --corpus_dir "/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/Corpus/library_source_code/version_corpus"