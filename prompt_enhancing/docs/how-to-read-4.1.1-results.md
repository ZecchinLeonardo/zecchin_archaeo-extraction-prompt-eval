# How to Read and Interpret 4.1.1 Notebook Results

## Quick Start Guide

The 4.1.1 notebook produces several types of output that help you understand how well the archaeological data extraction is working. This guide shows you exactly what to look for and how to interpret each result.

## Main Result Types

### 1. Summary Scores Table

**What you'll see:**
```
Average Metric: 4.9 / 7 (70.0%)
Best score so far: 80.0
```

**What it means:**
- **70.0%**: Overall accuracy across all fields
- **4.9 / 7**: Successfully extracted 4.9 out of 7 fields on average
- **80.0**: Best performance achieved during optimization

### 2. Field-by-Field Results

**Example output:**
```
| Field Name | Predicted Value | Expected Value | Score |
|------------|----------------|----------------|-------|
| intervention_start_date | 1987-09-27 | 1987-09-27 | ✔️ [0.450] |
| university.Comune | Roma | Roma | ✔️ [0.500] |
| building.Data_Protocollo | 2015-06-02 | 2015-06-02 | ✔️ [0.500] |
```

**How to read this:**
- **✔️**: Extraction was correct
- **❌**: Extraction failed or was wrong
- **[0.450]**: Confidence score (0.0 = no confidence, 1.0 = perfect confidence)
- **Predicted vs Expected**: Shows what the AI found vs. what should have been found

### 3. Document Processing Results

**Example:**
```
`%% Relazione_assistenza.pdf | Page [1, 2] (['table']) %%`
{'giorno': 2, 'mese': 'Luglio', 'anno': 2015}
```

**Interpretation:**
- **Document**: `Relazione_assistenza.pdf` 
- **Pages processed**: Pages 1 and 2
- **Content type**: Found tables
- **Extracted data**: Day=2, Month=July, Year=2015

## Understanding the Visualization Dashboard

When you run:
```python
visualizator.init_complete_vizualisation_engine(detailed_results)
visualizator.run_display_server()
```

You get an interactive web dashboard (typically at `http://localhost:8050`) with several views:

### 1. **Overview Dashboard**
Shows aggregate metrics:
- **Overall accuracy percentage** (e.g., "Pipeline Accuracy: 72.3%")
- **Total documents processed** (e.g., "Documents: 45/50 successful")
- **Processing time statistics** (e.g., "Avg time per document: 23.4s")

### 2. **Field Performance Analysis**
Interactive charts showing:
```
Extraction Accuracy by Field Type:
├── intervention_start_date: ████████░░ 83%
├── university.Comune:       ██████░░░░ 67%  
├── building.Data_Protocollo:████░░░░░░ 45%
└── university.Sigla:        █████░░░░░ 52%
```

### 3. **Document Quality Heatmap**
Visual representation showing:
- **Green**: Successful extractions (>80% fields correct)
- **Yellow**: Partial success (50-80% fields correct)  
- **Red**: Failed extractions (<50% fields correct)

### 4. **Error Pattern Analysis**
Tables showing common failure reasons:
```
Common Extraction Failures:
├── "Date format not recognized": 12 occurrences
├── "Location not in thesaurus": 8 occurrences
├── "Missing information in source": 15 occurrences
└── "OCR quality too poor": 5 occurrences
```

### 5. **Sample Predictions View**
Shows actual examples with context:
```
Document: Relazione_assistenza.pdf | Page 2
Source text: "L'intervento è stato condotto dal 15 al 20 maggio 1987"
Extracted: {"start_date": "1987-05-15", "end_date": "1987-05-20"}
Confidence: 0.78
Status: ✓ Correct
```

## Real Examples from the Notebook

### Example 1: Successful Date Extraction

**Input document text:**
```
"L'intervento archeologico è stato condotto dal 27 settembre 1987..."
```

**Result:**
```
intervention_start_date_min: 1987-09-27
intervention_start_date_max: 1987-09-27
intervention_start_date_precision: "day"
Score: ✔️ [0.450]
```

**What this means:**
- ✅ **Successfully extracted** the intervention date
- ✅ **Correct format** (YYYY-MM-DD)
- ✅ **Precise to the day** (not just month/year)
- ⚠️ **Medium confidence** (0.450) - could be improved

### Example 2: Location Extraction

**Input document text:**
```
"Scavi archeologici nel comune di Roma, provincia di Roma (RM)"
```

**Result:**
```
university.Comune: "Roma"
university.Sigla: "RM"
Score: ✔️ [0.500]
```

**What this means:**
- ✅ **Correctly identified** Rome as the municipality
- ✅ **Correctly extracted** RM as the province code
- ✅ **Good confidence** (0.500) for this type of extraction

### Example 3: Failed Extraction

**Input document text:**
```
"Il documento è stato archiviato in data imprecisata"
```

**Result:**
```
building.Data_Protocollo: None
Expected: "2015-06-02"
Score: ❌ [0.000]
```

**What this means:**
- ❌ **Failed to extract** the archiving date
- ❌ **Missing information** in the source text
- 💡 **Manual review needed** for this document

## Performance Benchmarks

### Excellent Performance (80%+ accuracy)
- **Structured dates** in standard formats
- **Clear location names** mentioned explicitly
- **Well-formatted documents** with clean OCR

### Good Performance (60-80% accuracy)  
- **Dates with variations** (e.g., "fine luglio 1987")
- **Location names** requiring inference
- **Documents with minor OCR errors**

### Poor Performance (<60% accuracy)
- **Handwritten documents** with poor OCR
- **Incomplete information** in source
- **Highly technical or abbreviated text**

## Troubleshooting Common Issues

### Issue 1: Low Overall Scores
**Symptoms:** Many ❌ results, scores below 0.3
**Possible causes:**
- Poor document quality (blurry scans, handwriting)
- LLM model not optimized for Italian text
- Training data mismatch

**Solutions:**
- Check document quality in the source PDFs
- Retrain with more representative examples
- Adjust confidence thresholds

### Issue 2: Inconsistent Results
**Symptoms:** Same type of information sometimes works, sometimes doesn't
**Possible causes:**
- Variable document formats
- Context window limitations
- Prompt optimization needed

**Solutions:**
- Review the `_focus_and_truncate()` function
- Run more DSPy optimization iterations
- Add more diverse training examples

### Issue 3: Missing Fields
**Symptoms:** Many `None` or `NaN` results
**Possible causes:**
- Information genuinely not present in documents
- OCR missed the relevant text sections
- Extraction prompts not finding the right patterns

**Solutions:**
- Manually verify a few documents to check if info exists
- Improve OCR preprocessing
- Refine the extraction prompts

## MLFlow Experiment Tracking

### Viewing Results in MLFlow

1. **Open MLFlow UI** (usually at `http://localhost:8887`)
2. **Find your experiment** (named something like "fresh-abc123")
3. **Compare runs** to see improvement over time

### Key Metrics to Track

- **Overall accuracy**: How well the entire pipeline works
- **Per-field accuracy**: Which fields need improvement
- **Processing time**: How long each document takes
- **Model parameters**: Which settings work best

### Comparing Experiments

Look for:
- **Accuracy trends** over different runs
- **Best performing configurations**
- **Trade-offs** between speed and accuracy

## Next Steps for Analysis

### 1. **Quality Assessment**
- Review low-scoring extractions manually
- Identify patterns in failures
- Plan targeted improvements

### 2. **Production Readiness**
- Set confidence thresholds for auto-acceptance
- Design manual review workflows for low-confidence results
- Plan batch processing strategies

### 3. **Continuous Improvement**
- Add new training examples based on failures
- Experiment with different LLM models
- Optimize prompts for specific document types

This guide should help you understand exactly what the 4.1.1 notebook is telling you about your archaeological data extraction pipeline's performance.