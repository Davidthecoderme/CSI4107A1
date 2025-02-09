import jsonlines

input_file = "scifact/corpus.jsonl"
output_file = "test_corpus.jsonl"

# Save the first 10 records to a new file
with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
    for i, obj in enumerate(reader):
        writer.write(obj)  # Write to the new file
        if i == 9:  # Stop after saving the first 10 records
            break

print(f"✅ The first 10 records have been saved to {output_file}")
