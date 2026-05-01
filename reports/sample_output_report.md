# Decision Intelligence Report

## Executive Summary
The sample analysis identifies churn risk patterns and next-month revenue signals from customer operating data.

## Model Performance
- Classification accuracy: 0.9
- Regression RMSE: 288.52

## Key Insights
- Higher support ticket volume and lower usage scores are strong churn-risk signals in the sample data.
- Enterprise customers with longer tenure show stronger next-month revenue expectations.
- Discount rate appears in both risk and revenue models, making it a useful monitoring field.

## Risk Explanation
- Risk level: moderate

### Main Drivers
- support_tickets
- usage_score
- tenure_months

### Limitations
Feature importance shows model signal strength, not causality.

## Recommendations
- Prioritize proactive retention outreach for customers with high ticket volume and low usage.
- Review discounting strategy for accounts with high churn risk and weak usage engagement.
- Add a weekly risk review using the top classification and revenue drivers.

## Expected Business Value
Improved intervention focus, better revenue visibility, and more consistent account prioritization.

## Measurement Plan
Track churn outcomes, revenue forecast error, and intervention conversion over the next 30 days.

