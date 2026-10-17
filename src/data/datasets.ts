import { DatasetInfo, DatasetSample } from '../types/flTypes';

// Helper PRNG for synthetic reproducible sample generation
function pseudoRandom(seed: number) {
  const x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

// Generate realistic Heart Disease Dataset
function generateHeartDiseaseData(): DatasetSample[] {
  const samples: DatasetSample[] = [];
  const count = 300;
  for (let i = 0; i < count; i++) {
    const age = 35 + Math.floor(pseudoRandom(i * 1.1) * 40); // 35 - 75
    const trestbps = 100 + Math.floor(pseudoRandom(i * 1.3) * 80); // 100 - 180
    const chol = 150 + Math.floor(pseudoRandom(i * 1.7) * 200); // 150 - 350
    const thalach = 90 + Math.floor(pseudoRandom(i * 1.9) * 100); // 90 - 190
    const oldpeak = parseFloat((pseudoRandom(i * 2.3) * 4.5).toFixed(1)); // 0 - 4.5
    const cp = Math.floor(pseudoRandom(i * 2.7) * 4); // 0-3 chest pain
    const fbs = pseudoRandom(i * 3.1) > 0.85 ? 1 : 0;
    const exang = pseudoRandom(i * 3.7) > 0.6 ? 1 : 0;
    
    // Disease risk probability calculation based on clinical markers
    const riskScore = (age - 50) * 0.03 + (chol - 200) * 0.008 + (trestbps - 120) * 0.01 + (160 - thalach) * 0.015 + oldpeak * 0.4 + exang * 0.8 + cp * 0.5;
    const label = riskScore > 0.5 ? 1 : 0;

    samples.push({
      id: i + 1,
      features: [age, trestbps, chol, thalach, oldpeak, cp, fbs, exang],
      label
    });
  }
  return samples;
}

// Generate PIMA Diabetes Dataset
function generateDiabetesData(): DatasetSample[] {
  const samples: DatasetSample[] = [];
  const count = 320;
  for (let i = 0; i < count; i++) {
    const pregnancies = Math.floor(pseudoRandom(i * 4.1) * 12);
    const glucose = 70 + Math.floor(pseudoRandom(i * 4.3) * 130);
    const bp = 50 + Math.floor(pseudoRandom(i * 4.7) * 60);
    const skin = 10 + Math.floor(pseudoRandom(i * 5.1) * 45);
    const insulin = 20 + Math.floor(pseudoRandom(i * 5.3) * 300);
    const bmi = parseFloat((18 + pseudoRandom(i * 5.7) * 25).toFixed(1));
    const dpf = parseFloat((0.1 + pseudoRandom(i * 6.1) * 1.5).toFixed(2));
    const age = 21 + Math.floor(pseudoRandom(i * 6.7) * 50);

    const risk = (glucose - 100) * 0.02 + (bmi - 25) * 0.06 + (age - 30) * 0.02 + dpf * 0.8 + insulin * 0.002;
    const label = risk > 0.6 ? 1 : 0;

    samples.push({
      id: i + 1,
      features: [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age],
      label
    });
  }
  return samples;
}

// Generate Breast Cancer Wisconsin Diagnostic Dataset
function generateBreastCancerData(): DatasetSample[] {
  const samples: DatasetSample[] = [];
  const count = 300;
  for (let i = 0; i < count; i++) {
    const radius = parseFloat((8.0 + pseudoRandom(i * 7.1) * 18.0).toFixed(2));
    const texture = parseFloat((10.0 + pseudoRandom(i * 7.3) * 25.0).toFixed(2));
    const perimeter = parseFloat((50.0 + radius * 6.28 + pseudoRandom(i * 7.7) * 15.0).toFixed(2));
    const area = parseFloat((200.0 + radius * radius * 3.14 + pseudoRandom(i * 7.9) * 100.0).toFixed(1));
    const smoothness = parseFloat((0.05 + pseudoRandom(i * 8.3) * 0.1).toFixed(4));
    const compactness = parseFloat((0.02 + pseudoRandom(i * 8.9) * 0.25).toFixed(4));
    const concavity = parseFloat((0.01 + pseudoRandom(i * 9.3) * 0.35).toFixed(4));
    const symmetry = parseFloat((0.1 + pseudoRandom(i * 9.7) * 0.2).toFixed(4));

    const risk = (radius - 14) * 0.3 + (perimeter - 90) * 0.05 + (area - 600) * 0.005 + (compactness - 0.1) * 5.0 + (concavity - 0.08) * 4.0;
    const label = risk > 0.2 ? 1 : 0; // 1: Malignant, 0: Benign

    samples.push({
      id: i + 1,
      features: [radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry],
      label
    });
  }
  return samples;
}

// Generate Hospital Readmission Dataset
function generateHospitalReadmissionData(): DatasetSample[] {
  const samples: DatasetSample[] = [];
  const count = 350;
  for (let i = 0; i < count; i++) {
    const age = 20 + Math.floor(pseudoRandom(i * 10.1) * 70);
    const comorbidities = Math.floor(pseudoRandom(i * 10.3) * 6);
    const priorAdmissions = Math.floor(pseudoRandom(i * 10.7) * 8);
    const lengthOfStay = 1 + Math.floor(pseudoRandom(i * 11.1) * 14);
    const labRiskScore = parseFloat((1.0 + pseudoRandom(i * 11.3) * 9.0).toFixed(1));
    const emergencyVisits = Math.floor(pseudoRandom(i * 11.7) * 5);

    const readmitProb = (age - 60) * 0.02 + comorbidities * 0.4 + priorAdmissions * 0.3 + lengthOfStay * 0.1 + labRiskScore * 0.25;
    const label = readmitProb > 2.5 ? 1 : 0;

    samples.push({
      id: i + 1,
      features: [age, comorbidities, priorAdmissions, lengthOfStay, labRiskScore, emergencyVisits],
      label
    });
  }
  return samples;
}

export const BUILTIN_DATASETS: DatasetInfo[] = [
  {
    id: 'heart_disease',
    name: 'Cleveland Heart Disease Clinical Dataset',
    description: 'Multi-institutional cardiac health diagnostic dataset with 8 patient clinical parameters.',
    sampleCount: 300,
    featureNames: ['Age', 'Resting_BP', 'Cholesterol', 'Max_Heart_Rate', 'ST_Depression', 'Chest_Pain_Type', 'Fasting_BS', 'Exercise_Angina'],
    targetName: 'Heart_Disease_Risk',
    data: generateHeartDiseaseData()
  },
  {
    id: 'diabetes_pima',
    name: 'PIMA Indian Diabetes Dataset',
    description: 'Metabolic & endocrinology diagnostic metric records for diabetes classification.',
    sampleCount: 320,
    featureNames: ['Pregnancies', 'Glucose', 'Blood_Pressure', 'Skin_Thickness', 'Insulin', 'BMI', 'Diabetes_Pedigree', 'Age'],
    targetName: 'Diabetes_Positive',
    data: generateDiabetesData()
  },
  {
    id: 'breast_cancer',
    name: 'Wisconsin Diagnostic Breast Cancer (WDBC)',
    description: 'Oncology cell nuclear feature measurements derived from fine needle aspirate (FNA) images.',
    sampleCount: 300,
    featureNames: ['Radius_Mean', 'Texture_Mean', 'Perimeter_Mean', 'Area_Mean', 'Smoothness', 'Compactness', 'Concavity', 'Symmetry'],
    targetName: 'Malignancy_Status',
    data: generateBreastCancerData()
  },
  {
    id: 'hospital_readmission',
    name: 'Healthcare Insurance & Hospital 30-Day Readmission Risk',
    description: 'Electronic Health Record (EHR) longitudinal patient readmission & comorbidity factors.',
    sampleCount: 350,
    featureNames: ['Patient_Age', 'Comorbidities_Count', 'Prior_Admissions', 'Length_of_Stay_Days', 'Lab_Risk_Score', 'Emergency_Visits'],
    targetName: 'Readmission_Risk_30d',
    data: generateHospitalReadmissionData()
  }
];

export function parseCSVData(csvText: string, targetColName?: string): {
  data: DatasetSample[];
  featureNames: string[];
  targetName: string;
  detectedColumns: string[];
} {
  const lines = csvText.trim().split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length < 2) {
    throw new Error('CSV file must contain at least a header row and one data row.');
  }

  const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
  let targetIndex = targetColName ? headers.indexOf(targetColName) : headers.length - 1;
  if (targetIndex === -1) targetIndex = headers.length - 1;

  const targetName = headers[targetIndex];
  const featureNames = headers.filter((_, idx) => idx !== targetIndex);

  const samples: DatasetSample[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim());
    if (values.length !== headers.length) continue;

    const rawTarget = values[targetIndex];
    let label = parseFloat(rawTarget);
    if (isNaN(label)) {
      // Categorical fallback
      const lower = rawTarget.toLowerCase();
      label = (lower === 'true' || lower === '1' || lower === 'yes' || lower === 'malignant' || lower === 'positive' || lower === 'high') ? 1 : 0;
    } else {
      label = label > 0 ? 1 : 0;
    }

    const features: number[] = [];
    headers.forEach((_, idx) => {
      if (idx !== targetIndex) {
        const val = parseFloat(values[idx]);
        features.push(isNaN(val) ? 0 : val);
      }
    });

    samples.push({
      id: i,
      features,
      label
    });
  }

  return {
    data: samples,
    featureNames,
    targetName,
    detectedColumns: headers
  };
}
