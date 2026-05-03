import datetime

def generate_text_report(diagnostics, expert_report, orig_results, cleaned_results, task_type, selected_algo):
    """
    Generates a structured text report summarizing the dataset diagnostics, expert analysis, and ML results.
    This function compiles all findings into a professional document.
    """
    report = []
    # --- BLOCK 1: REPORT HEADER ---
    # We add the title, project name, and current timestamp to the report.
    report.append("="*70)
    report.append(" PROFESSIONAL DATASET ASSESSMENT & AutoML REPORT")
    report.append(" FYP: Automated Dataset Diagnostics and Repair Framework")
    report.append(" Generated on: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report.append("="*70 + "\n")

    # --- BLOCK 2: EXECUTIVE SUMMARY ---
    # We show the high-level details: is the data suitable? what is the quality score?
    report.append("1. EXECUTIVE SUMMARY & SUITABILITY")
    report.append("-" * 40)
    summary = expert_report['summary']
    report.append(f"Version:             {summary['version']}")
    report.append(f"Suitability Status:  {summary['suitability']}")
    report.append(f"Overall Quality Score: {summary['quality_score']}/100")
    report.append(f"Mapped Domain:       {summary['domain']}")
    report.append(f"Primary Industry:    {summary['industry']}")
    report.append(f"Dataset Dimensions:  {summary['rows']} rows x {summary['cols']} columns")
    report.append("\n")

    # --- BLOCK 3: ISSUE IDENTIFICATION (Severity) ---
    # We list all identified issues along with their severity (High/Medium).
    report.append("2. ISSUE IDENTIFICATION & CLASSIFICATION")
    report.append("-" * 40)
    if diagnostics.get('identified_issues'):
        for issue in diagnostics['identified_issues']:
            report.append(f"  [!] {issue['type']:<20} | Severity: {issue['severity']:<10} | Impact Score: {issue['score']:.4f}")
    else:
        report.append("  [+] No critical data quality issues identified.")
    report.append("\n")

    # --- BLOCK 4: INTERVENTION MODULE ---
    # We list the repair strategies we chose and why we chose them.
    report.append("3. INTERVENTION SELECTION MODULE")
    report.append("-" * 40)
    if expert_report.get('interventions'):
        for item in expert_report['interventions']:
            report.append(f"  Issue:     {item['issue']}")
            report.append(f"  Strategy:  {item['strategy']}")
            report.append(f"  Rationale: {item['rationale']}")
            report.append("-" * 20)
    else:
        report.append("  No interventions required.")
    report.append("\n")

    # --- BLOCK 5: SWOT ANALYSIS ---
    # We provide a simple list of the dataset's Strengths and Weaknesses.
    report.append("4. SWOT ANALYSIS (Strengths & Weaknesses)")
    report.append("-" * 40)
    report.append("STRENGTHS:")
    for s in expert_report['swot']['strengths']: report.append(f"  [+] {s}")
    report.append("\nWEAKNESSES:")
    for w in expert_report['swot']['weaknesses']: report.append(f"  [-] {w}")
    report.append("\n")

    # --- BLOCK 6: DETAILED DIAGNOSTICS ---
    # We provide the raw numbers found during analysis (duplicates, noise, missing values).
    report.append("5. DETAILED DATA DIAGNOSTICS")
    report.append("-" * 40)
    report.append(f"Total Duplicates: {diagnostics['duplicates']}")
    report.append(f"Total Missing Values: {diagnostics['missing_values']['total']}")
    report.append(f"Total Outliers: {diagnostics['outliers']['total']}")
    report.append(f"IsolationForest Outliers: {diagnostics.get('outliers', {}).get('isolation_forest', 0)}")
    report.append(f"Label Noise: {diagnostics.get('label_noise', 0)} samples")
    report.append(f"Label Noise Method: {diagnostics.get('label_noise_method', 'n/a')}")
    report.append(f"Near-Duplicates (semantic): {diagnostics.get('near_duplicates', 0)}")
    report.append(f"Target Column: {diagnostics['class_imbalance']['target_column']}")
    
    # Add mixed fields information
    if diagnostics.get('mixed_fields'):
        report.append(f"\nMixed Field Inconsistencies: {len(diagnostics['mixed_fields'])} columns")
        for col, info in diagnostics['mixed_fields'].items():
            report.append(f"  - {col}: {info['type']} ({info.get('numeric_count', info.get('date_count', 0))} valid, {info.get('text_count', info.get('non_date_count', 0))} invalid)")
    
    if diagnostics.get('correlations'):
        report.append("\nTop Feature Correlations with Target:")
        for feat, val in diagnostics['correlations'].items():
            report.append(f"  - {feat}: {val}")
    if diagnostics.get('spearman_correlations'):
        report.append("\nTop Spearman Correlations with Target:")
        for feat, val in diagnostics['spearman_correlations'].items():
            report.append(f"  - {feat}: {val}")
    if diagnostics.get('mutual_information'):
        report.append("\nMutual Information with Target:")
        for feat, val in diagnostics['mutual_information'].items():
            report.append(f"  - {feat}: {val}")

    if diagnostics.get('distribution_skew', {}).get('skewed_features'):
        report.append("\nDistribution Skew (|skew| > 1):")
        for feat, val in diagnostics['distribution_skew']['skewed_features'].items():
            report.append(f"  - {feat}: {val}")
    
    if diagnostics.get('leakage_risk'):
        report.append(f"\nPotential Data Leakage Risk: {', '.join(diagnostics['leakage_risk'])}")

    report.append("\nClass Distribution:")
    for cls, count in diagnostics['class_imbalance']['distribution'].items():
        report.append(f"  - Class {cls}: {count}")
    report.append("\n")

    # --- BLOCK 7: PERFORMANCE COMPARISON ---
    # We show the "Before" vs "After" metrics for each model to prove 
    # that our repairs actually improved the results.
    report.append("6. MACHINE LEARNING PERFORMANCE COMPARISON")
    report.append("-" * 40)
    report.append(f"Task Type: {task_type.capitalize()}")
    report.append(f"Selected Algorithm: {selected_algo}")
    
    for model_name, metrics in cleaned_results.items():
        if 'error' in metrics:
            report.append(f"\nModel: {model_name}")
            report.append(f"  ERROR: {metrics['error']}")
        else:
            report.append(f"\nModel: {model_name}")
            orig_metrics = orig_results.get(model_name, {})
            
            if task_type == 'classification':
                report.append(f"  Metric       | Original | Repaired")
                report.append(f"  Accuracy     | {orig_metrics.get('Accuracy', 'N/A'):<8} | {metrics['Accuracy']}")
                report.append(f"  Gen Gap      | {orig_metrics.get('Generalization Gap', 'N/A'):<8} | {metrics.get('Generalization Gap', 'N/A')}")
                report.append(f"  Precision    | {orig_metrics.get('Precision', 'N/A'):<8} | {metrics['Precision']}")
                report.append(f"  Recall       | {orig_metrics.get('Recall', 'N/A'):<8} | {metrics['Recall']}")
                report.append(f"  F1-Score     | {orig_metrics.get('F1-Score', 'N/A'):<8} | {metrics['F1-Score']}")
            else:
                report.append(f"  Metric       | Original | Repaired")
                report.append(f"  R2 Score     | {orig_metrics.get('R2 Score', 'N/A'):<8} | {metrics['R2 Score']}")
                report.append(f"  Gen Gap      | {orig_metrics.get('Generalization Gap', 'N/A'):<8} | {metrics.get('Generalization Gap', 'N/A')}")
                report.append(f"  MAE          | {orig_metrics.get('MAE', 'N/A'):<8} | {metrics.get('MAE', 'N/A')}")
            
            if 'feature_importance' in metrics:
                report.append("  Top Features (Repaired):")
                sorted_features = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
                for feat, imp in sorted_features:
                    report.append(f"    - {feat}: {imp}")

    # --- BLOCK 8: FINAL DETERMINATION ---
    # We end with a clear approval or rejection of the dataset for production use.
    report.append("\n" + "="*70)
    report.append(" FINAL ELIGIBILITY DETERMINATION")
    if "HIGHLY SUITABLE" in summary['suitability']:
        report.append(" Dataset is APPROVED for high-stakes production use cases.")
    elif "CONDITIONALLY SUITABLE" in summary['suitability']:
        report.append(" Dataset is ELIGIBLE for experimental use; follow recommendations.")
    else:
        report.append(" Dataset is NOT SUITABLE for current project objectives.")
    report.append("="*70)

    # Join all lines into a single string for the final .txt file
    return "\n".join(report)
