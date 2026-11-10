-- PeoplePulse HR Intelligence: PostgreSQL portfolio queries
-- Assumes CSVs have been imported into employees, performance_reviews,
-- monthly_attendance, training_records and recruitment_pipeline.

-- 1. Department workforce and attrition summary
SELECT department, COUNT(*) AS headcount,
       SUM(attrition_flag) AS exits,
       ROUND(100.0 * SUM(attrition_flag) / NULLIF(COUNT(*),0), 2) AS attrition_pct,
       ROUND(AVG(tenure_years)::numeric, 2) AS avg_tenure
FROM employees GROUP BY department ORDER BY headcount DESC;

-- 2. Attrition by tenure band using a CTE
WITH tenure_bands AS (
  SELECT employee_id, attrition_flag,
    CASE WHEN tenure_years < 1 THEN '<1 year'
         WHEN tenure_years < 3 THEN '1–3 years'
         WHEN tenure_years < 5 THEN '3–5 years'
         ELSE '5+ years' END AS tenure_band
  FROM employees
)
SELECT tenure_band, COUNT(*) AS employees,
       ROUND(100.0 * SUM(attrition_flag) / COUNT(*), 2) AS attrition_pct
FROM tenure_bands GROUP BY tenure_band ORDER BY MIN(
  CASE tenure_band WHEN '<1 year' THEN 1 WHEN '1–3 years' THEN 2
                   WHEN '3–5 years' THEN 3 ELSE 4 END);

-- 3. Latest performance review per employee (window function)
WITH ranked_reviews AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY review_year DESC) AS rn
  FROM performance_reviews
)
SELECT e.employee_id, e.department, r.review_year, r.performance_rating,
       r.goal_completion_pct
FROM employees e JOIN ranked_reviews r USING (employee_id)
WHERE r.rn = 1;

-- 4. Attendance trends by month and department
SELECT a.month, e.department,
       ROUND(AVG(a.days_absent)::numeric, 2) AS avg_absence_days,
       ROUND(AVG(a.late_arrivals)::numeric, 2) AS avg_late_arrivals
FROM monthly_attendance a JOIN employees e USING (employee_id)
GROUP BY a.month, e.department ORDER BY a.month, e.department;

-- 5. Training completion and average assessment by category
SELECT course_category, COUNT(*) AS enrollments,
       COUNT(*) FILTER (WHERE completion_status='Completed') AS completions,
       ROUND(100.0 * COUNT(*) FILTER (WHERE completion_status='Completed') /
             NULLIF(COUNT(*),0), 2) AS completion_pct,
       ROUND(AVG(assessment_score)::numeric, 1) AS avg_score
FROM training_records GROUP BY course_category ORDER BY completion_pct DESC;

-- 6. Recruitment source conversion
SELECT candidate_source, COUNT(*) AS applications,
       COUNT(*) FILTER (WHERE current_stage='Hired') AS hires,
       ROUND(100.0 * COUNT(*) FILTER (WHERE current_stage='Hired') /
             NULLIF(COUNT(*),0), 2) AS hire_conversion_pct,
       ROUND(AVG(time_to_hire_days)::numeric, 1) AS avg_time_to_hire
FROM recruitment_pipeline GROUP BY candidate_source
ORDER BY hire_conversion_pct DESC;

-- 7. Employee 360° join
SELECT e.employee_id, e.department, e.job_level, e.location,
       e.tenure_years, e.satisfaction_score, e.engagement_score,
       p.review_year, p.performance_rating, p.goal_completion_pct,
       t.course_category, t.completion_status, t.assessment_score
FROM employees e
LEFT JOIN performance_reviews p USING (employee_id)
LEFT JOIN training_records t USING (employee_id)
WHERE e.employee_id = 'EMP00001'
ORDER BY p.review_year DESC, t.enrollment_date DESC;
