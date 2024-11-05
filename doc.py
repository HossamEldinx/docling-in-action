from docling.document_converter import DocumentConverter

source = "docling.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)

# Print results to console
print(result.document.export_to_markdown())

print(result.document.export_to_dict()) # export to JSON

# Save results to a text file
with open('conversion_results.txt', 'w', encoding='utf-8') as f:
    f.write("\n\nMarkdown Conversion:\n")
    f.write(result.document.export_to_markdown())
