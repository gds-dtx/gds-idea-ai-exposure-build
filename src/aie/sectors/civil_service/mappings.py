"""Civil Service mapping tables: GRID department, profession and grade labels -> Civil Service
Statistics categories, the statistics download URLs and the raking thresholds.

Copied from gds-idea-ai-exposure (src/constants.py at HEAD, commit e02cbd4) so the build no
longer depends on that repo; paths, AWS settings and model lists were left out.

The GRID cleaning step of release v2 was run with the tables as they were at commit e246377
(before the 2026 statistics update), when two since-abolished bodies were still listed and one
multi-grade label was marked invalid. Those differences are CLEANING_DEPARTMENTS and
CLEANING_GRADE_MAPPING at the end, so v2's cleaning reproduces exactly.
"""

from __future__ import annotations

# Departments
DEPARTMENTS = {
    "active-travel-england": {
        "dept_key": "Active Travel England",
        "grid_name": "Active Travel England",
        "cs_bulletin_org_name": "Active Travel England",
        "parent_dept": "Department for Transport",
    },
    "advisory-conciliation-and-arbitration-service": {
        "dept_key": "ACAS",
        "grid_name": "Advisory, Conciliation and Arbitration Service",
        "cs_bulletin_org_name": "Advisory, Conciliation and Arbitration Service",
        "parent_dept": "Department for Business and Trade",
    },
    "animal-and-plant-health-agency": {
        "dept_key": "APHA",
        "grid_name": "Animal and Plant Health Agency",
        "cs_bulletin_org_name": "Animal and Plant Health Agency",
        "parent_dept": "Department for Environment, Food and Rural Affairs",
    },
    "attorney-generals-office": {
        "dept_key": "AGO",
        "grid_name": "Attorney General's Office",
        "cs_bulletin_org_name": "Attorney General's Office",
        "parent_dept": "Attorney General's Departments",
    },
    "building-digital-uk-bduk": {
        "dept_key": "BDUK",
        "grid_name": "Building Digital UK (BDUK)",
        "cs_bulletin_org_name": "Building Digital UK",
        "parent_dept": "Department for Science, Innovation and Technology",
    },
    "cabinet-office": {
        "dept_key": "CO",
        "grid_name": "Cabinet Office",
        "cs_bulletin_org_name": "Cabinet Office (excl. agencies)",
        "parent_dept": "Cabinet Office",
    },
    "centre-for-environment-fisheries-and-aquaculture-science": {
        "dept_key": "CEFAS",
        "grid_name": "Centre for Environment, Fisheries and Aquaculture Science",
        "cs_bulletin_org_name": "Centre for Environment, Fisheries and Aquaculture Science",
        "parent_dept": "Department for Environment, Food and Rural Affairs",
    },
    "charity-commission": {
        "dept_key": "Charity Commission",
        "grid_name": "Charity Commission",
        "cs_bulletin_org_name": "Charity Commission",
        "parent_dept": "Charity Commission",
    },
    "companies-house": {
        "dept_key": "Companies House",
        "grid_name": "Companies House",
        "cs_bulletin_org_name": "Companies House",
        "parent_dept": "Department for Business and Trade",
    },
    "competition-markets-authority": {
        "dept_key": "CMA",
        "grid_name": "Competition & Markets Authority",
        "cs_bulletin_org_name": "Competition and Markets Authority",
        "parent_dept": "Competition and Markets Authority",
    },
    "consumer-scotland": {
        "dept_key": "Consumer Scotland",
        "grid_name": "Consumer Scotland",
        "cs_bulletin_org_name": "Consumer Scotland",
        "parent_dept": "Scottish Government",
    },
    "criminal-injuries-compensation-authority": {
        "dept_key": "CICA",
        "grid_name": "Criminal Injuries Compensation Authority",
        "cs_bulletin_org_name": "Criminal Injuries Compensation Authority",
        "parent_dept": "Ministry of Justice",
    },
    "crown-commercial-service": {
        "dept_key": "CCS",
        "grid_name": "Crown Commercial Service",
        "cs_bulletin_org_name": "Crown Commercial Service",
        "parent_dept": "Cabinet Office",
    },
    "crown-office-and-procurator-fiscal-service": {
        "dept_key": "COPFS",
        "grid_name": "Crown Office and Procurator Fiscal Service",
        "cs_bulletin_org_name": "Crown Office and Procurator Fiscal Service",
        "parent_dept": "Scottish Government",
    },
    "crown-prosecution-service": {
        "dept_key": "CPS",
        "grid_name": "Crown Prosecution Service",
        "cs_bulletin_org_name": "Crown Prosecution Service",
        "parent_dept": "Attorney General's Departments",
    },
    "defence-science-and-technology-laboratory": {
        "dept_key": "DSTL",
        "grid_name": "Defence Science and Technology Laboratory",
        "cs_bulletin_org_name": "Defence Science and Technology Laboratory",
        "parent_dept": "Ministry of Defence",
    },
    "department-for-business-and-trade": {
        "dept_key": "DBT",
        "grid_name": "Department for Business and Trade",
        "cs_bulletin_org_name": "Department for Business and Trade (excl. agencies)",
        "parent_dept": "Department for Business and Trade",
    },
    "department-for-culture-media-and-sport": {
        "dept_key": "DCMS",
        "grid_name": "Department for Culture, Media and Sport",
        "cs_bulletin_org_name": "Department for Culture, Media and Sport",
        "parent_dept": "Department for Culture, Media and Sport",
    },
    "department-for-education": {
        "dept_key": "DfE",
        "grid_name": "Department for Education",
        "cs_bulletin_org_name": "Department for Education (excl. agencies)",
        "parent_dept": "Department for Education",
    },
    "department-for-energy-security-net-zero": {
        "dept_key": "DESNZ",
        "grid_name": "Department for Energy Security & Net Zero",
        "cs_bulletin_org_name": "Department for Energy Security and Net Zero",
        "parent_dept": "Department for Energy Security and Net Zero",
    },
    "department-for-environment-food-and-rural-affairs": {
        "dept_key": "Defra",
        "grid_name": "Department for Environment, Food and Rural Affairs",
        "cs_bulletin_org_name": "Department for Environment, Food and Rural Affairs (excl. agencies)",
        "parent_dept": "Department for Environment, Food and Rural Affairs",
    },
    "department-for-science-innovation-and-technology": {
        "dept_key": "DSIT",
        "grid_name": "Department for Science, Innovation and Technology",
        "cs_bulletin_org_name": "Department for Science, Innovation and Technology (excl. agencies)",
        "parent_dept": "Department for Science, Innovation and Technology",
    },
    "department-for-transport": {
        "dept_key": "DfT",
        "grid_name": "Department for Transport",
        "cs_bulletin_org_name": "Department for Transport (excl. agencies)",
        "parent_dept": "Department for Transport",
    },
    "department-for-work-and-pensions": {
        "dept_key": "DWP",
        "grid_name": "Department for Work and Pensions",
        "cs_bulletin_org_name": "Department for Work and Pensions (excl. agencies)",
        "parent_dept": "Department for Work and Pensions",
    },
    "department-of-health-and-social-care": {
        "dept_key": "DHSC",
        "grid_name": "Department of Health and Social Care",
        "cs_bulletin_org_name": "Department of Health and Social Care (excl. agencies)",
        "parent_dept": "Department of Health and Social Care",
    },
    "driver-and-vehicle-licensing-agency": {
        "dept_key": "DVLA",
        "grid_name": "Driver and Vehicle Licensing Agency",
        "cs_bulletin_org_name": "Driver and Vehicle Licensing Agency",
        "parent_dept": "Department for Transport",
    },
    "driver-and-vehicle-standards-agency": {
        "dept_key": "DVSA",
        "grid_name": "Driver and Vehicle Standards Agency",
        "cs_bulletin_org_name": "Driver and Vehicle Standards Agency",
        "parent_dept": "Department for Transport",
    },
    "estyn": {
        "dept_key": "ESTYN",
        "grid_name": "ESTYN",
        "cs_bulletin_org_name": "Estyn",
        "parent_dept": "Estyn",
    },
    "fcdo-services": {
        "dept_key": "FCDO Services",
        "grid_name": "FCDO Services",
        "cs_bulletin_org_name": "FCDO Services",
        "parent_dept": "Foreign, Commonwealth and Development Office",
    },
    "fast-stream-and-emerging-talent": {
        "dept_key": "Fast Stream",
        "grid_name": "Fast Stream and Emerging Talent",
        "cs_bulletin_org_name": "Central Civil Service Fast Stream",
        "parent_dept": "Cabinet Office",
    },
    "food-standards-agency": {
        "dept_key": "FSA",
        "grid_name": "Food Standards Agency",
        "cs_bulletin_org_name": "Food Standards Agency",
        "parent_dept": "Food Standards Agency",
    },
    "foreign-commonwealth-development-office": {
        "dept_key": "FCDO",
        "grid_name": "Foreign, Commonwealth & Development Office",
        "cs_bulletin_org_name": "Foreign, Commonwealth and Development Office (excl. agencies)",
        "parent_dept": "Foreign, Commonwealth and Development Office",
    },
    "forestry-and-land-scotland": {
        "dept_key": "Forestry and Land Scotland",
        "grid_name": "Forestry and Land Scotland",
        "cs_bulletin_org_name": "Forestry and Land Scotland",
        "parent_dept": "Scottish Government",
    },
    "government-actuarys-department": {
        "dept_key": "GAD",
        "grid_name": "Government Actuary's Department",
        "cs_bulletin_org_name": "Government Actuary's Department",
        "parent_dept": "Chancellor's other departments",
    },
    "government-commercial-function": {
        "dept_key": "GCO",
        "grid_name": "Government Commercial Function",
        "cs_bulletin_org_name": "Government Commercial Organisation",
        "parent_dept": "Cabinet Office",
    },
    "government-internal-audit-agency": {
        "dept_key": "GIAA",
        "grid_name": "Government Internal Audit Agency",
        "cs_bulletin_org_name": "Government Internal Audit Agency",
        "parent_dept": "HM Treasury",
    },
    "government-legal-department": {
        "dept_key": "GLD",
        "grid_name": "Government Legal Department",
        "cs_bulletin_org_name": "Government Legal Department",
        "parent_dept": "Attorney General's Departments",
    },
    "government-property-agency": {
        "dept_key": "GPA",
        "grid_name": "Government Property Agency",
        "cs_bulletin_org_name": "Government Property Agency",
        "parent_dept": "Cabinet Office",
    },
    "hm-courts-and-tribunals-service": {
        "dept_key": "HMCTS",
        "grid_name": "HM Courts and Tribunals Service",
        "cs_bulletin_org_name": "HM Courts and Tribunals Service",
        "parent_dept": "Ministry of Justice",
    },
    "hm-crown-prosecution-service-inspectorate": {
        "dept_key": "HMCPSI",
        "grid_name": "HM Crown Prosecution Service Inspectorate",
        "cs_bulletin_org_name": "HM Crown Prosecution Service Inspectorate",
        "parent_dept": "Attorney General's Departments",
    },
    "hm-land-registry": {
        "dept_key": "HMLR",
        "grid_name": "HM Land Registry",
        "cs_bulletin_org_name": "HM Land Registry",
        "parent_dept": "HM Land Registry",
    },
    "hm-prison-probation-service": {
        "dept_key": "HMPPS",
        "grid_name": "HM Prison & Probation Service",
        "cs_bulletin_org_name": "HM Prison and Probation Service",
        "parent_dept": "Ministry of Justice",
    },
    "hm-revenue-and-customs": {
        "dept_key": "HMRC",
        "grid_name": "HM Revenue and Customs",
        "cs_bulletin_org_name": "HM Revenue and Customs (excl. agencies)",
        "parent_dept": "HM Revenue and Customs",
    },
    "hm-treasury": {
        "dept_key": "HMT",
        "grid_name": "HM Treasury",
        "cs_bulletin_org_name": "HM Treasury (excl. agencies)",
        "parent_dept": "HM Treasury",
    },
    "health-and-safety-executive": {
        "dept_key": "HSE",
        "grid_name": "Health and Safety Executive",
        "cs_bulletin_org_name": "Health and Safety Executive",
        "parent_dept": "Department for Work and Pensions",
    },
    "home-office": {
        "dept_key": "HO",
        "grid_name": "Home Office",
        "cs_bulletin_org_name": "Home Office",
        "parent_dept": "Home Office",
    },
    "insolvency-service": {
        "dept_key": "Insolvency Service",
        "grid_name": "Insolvency Service",
        "cs_bulletin_org_name": "The Insolvency Service",
        "parent_dept": "Department for Business and Trade",
    },
    # REMOVED IN 2026 CS STATS
    # 'institute-for-apprenticeships-and-technical-education':
    #     {'dept_key': 'IfATE',
    #      'grid_name': 'Institute for Apprenticeships and Technical Education',
    #      'cs_bulletin_org_name': 'Institute for Apprenticeships and Technical Education',
    #      'parent_dept': 'Department for Education'},
    "intellectual-property-office": {
        "dept_key": "IPO",
        "grid_name": "Intellectual Property Office",
        "cs_bulletin_org_name": "Intellectual Property Office",
        "parent_dept": "Department for Science, Innovation and Technology",
    },
    "legal-aid-agency": {
        "dept_key": "LAA",
        "grid_name": "Legal Aid Agency",
        "cs_bulletin_org_name": "Legal Aid Agency",
        "parent_dept": "Ministry of Justice",
    },
    "maritime-and-coastguard-agency": {
        "dept_key": "MCA",
        "grid_name": "Maritime and Coastguard Agency",
        "cs_bulletin_org_name": "Maritime and Coastguard Agency",
        "parent_dept": "Department for Transport",
    },
    "medicines-and-healthcare-products-regulatory-agency": {
        "dept_key": "MHRA",
        "grid_name": "Medicines and Healthcare Products Regulatory Agency",
        "cs_bulletin_org_name": "Medicines and Healthcare Products Regulatory Agency",
        "parent_dept": "Department of Health and Social Care",
    },
    "met-office": {
        "dept_key": "Met Office",
        "grid_name": "Met Office",
        "cs_bulletin_org_name": "Met Office",
        "parent_dept": "Department for Science, Innovation and Technology",
    },
    "ministry-of-defence": {
        "dept_key": "MoD",
        "grid_name": "Ministry of Defence",
        "cs_bulletin_org_name": "Ministry of Defence (excl. agencies)",
        "parent_dept": "Ministry of Defence",
    },
    "ministry-of-housing-communities-and-local-government": {
        "dept_key": "MHCLG",
        "grid_name": "Ministry of Housing, Communities and Local Government",
        "cs_bulletin_org_name": "Ministry of Housing, Communities and Local Government (excl. agencies)",
        "parent_dept": "Ministry of Housing, Communities and Local Government",
    },
    "ministry-of-justice": {
        "dept_key": "MoJ",
        "grid_name": "Ministry of Justice",
        "cs_bulletin_org_name": "Ministry of Justice (excl. agencies)",
        "parent_dept": "Ministry of Justice",
    },
    "national-crime-agency": {
        "dept_key": "NCA",
        "grid_name": "National Crime Agency",
        "cs_bulletin_org_name": "National Crime Agency",
        "parent_dept": "National Crime Agency",
    },
    # REMOVED IN 2026 UKCS
    # 'national-infrastructure-commission':
    #     {'dept_key': 'NIC',
    #      'grid_name': 'National Infrastructure Commission',
    #      'cs_bulletin_org_name': 'National Infrastructure Commission',
    #      'parent_dept': 'HM Treasury'},
    "national-savings-and-investments": {
        "dept_key": "NS&I",
        "grid_name": "National Savings and Investments",
        "cs_bulletin_org_name": "National Savings and Investments",
        "parent_dept": "Chancellor's other departments",
    },
    "northern-ireland-office": {
        "dept_key": "NIO",
        "grid_name": "Northern Ireland Office",
        "cs_bulletin_org_name": "Northern Ireland Office",
        "parent_dept": "Northern Ireland Office",
    },
    "ofgem": {
        "dept_key": "OFGEM",
        "grid_name": "OFGEM",
        "cs_bulletin_org_name": "Office of Gas and Electricity Markets",
        "parent_dept": "Office of Gas and Electricity Markets",
    },
    "office-for-budget-responsibility": {
        "dept_key": "OBR",
        "grid_name": "Office for Budget Responsibility",
        "cs_bulletin_org_name": "Office for Budget Responsibility",
        "parent_dept": "HM Treasury",
    },
    "office-for-standards-in-education-childrens-services-and-skills": {
        "dept_key": "Ofsted",
        "grid_name": "Office for Standards in Education, Children's Services and Skills",
        "cs_bulletin_org_name": "Office for Standards in Education, Children's Services and Skills",
        "parent_dept": "Office for Standards in Education, Children's Services and Skills",
    },
    "office-of-qualifications-and-examinations-regulation": {
        "dept_key": "Ofqual",
        "grid_name": "Office of Qualifications and Examinations Regulation",
        "cs_bulletin_org_name": "Office of Qualifications and Examinations Regulation",
        "parent_dept": "Office of Qualifications and Examinations Regulation",
    },
    "office-of-rail-and-road": {
        "dept_key": "ORR",
        "grid_name": "Office of Rail and Road",
        "cs_bulletin_org_name": "Office of Rail and Road",
        "parent_dept": "Office of Rail and Road",
    },
    "office-of-the-public-guardian": {
        "dept_key": "OPG",
        "grid_name": "Office of the Public Guardian",
        "cs_bulletin_org_name": "Office of the Public Guardian",
        "parent_dept": "Ministry of Justice",
    },
    "office-of-the-secretary-of-state-for-wales": {
        "dept_key": "OSSW",
        "grid_name": "Office of the Secretary of State for Wales",
        "cs_bulletin_org_name": "Office of the Secretary of State for Wales",
        "parent_dept": "Office of the Secretary of State for Wales",
    },
    "ofwat-water-services-regulation-authority": {
        "dept_key": "Ofwat",
        "grid_name": "Ofwat (Water Services Regulation Authority)",
        "cs_bulletin_org_name": "Water Services Regulation Authority",
        "parent_dept": "Water Services Regulation Authority",
    },
    "planning-inspectorate": {
        "dept_key": "Planning Inspectorate",
        "grid_name": "Planning Inspectorate",
        "cs_bulletin_org_name": "Planning Inspectorate",
        "parent_dept": "Ministry of Housing, Communities and Local Government",
    },
    "queen-elizabeth-ii-conference-centre": {
        "dept_key": "QEII Centre",
        "grid_name": "Queen Elizabeth II Conference Centre",
        "cs_bulletin_org_name": "Queen Elizabeth II Centre",
        "parent_dept": "Ministry of Housing, Communities and Local Government",
    },
    "registers-of-scotland": {
        "dept_key": "RoS",
        "grid_name": "Registers of Scotland",
        "cs_bulletin_org_name": "Registers of Scotland",
        "parent_dept": "Scottish Government",
    },
    "rural-payments-agency": {
        "dept_key": "RPA",
        "grid_name": "Rural Payments Agency",
        "cs_bulletin_org_name": "Rural Payments Agency",
        "parent_dept": "Department for Environment, Food and Rural Affairs",
    },
    "scottish-forestry": {
        "dept_key": "Scottish Forestry",
        "grid_name": "Scottish Forestry",
        "cs_bulletin_org_name": "Scottish Forestry",
        "parent_dept": "Scottish Government",
    },
    "scottish-government": {
        "dept_key": "SG",
        "grid_name": "Scottish Government",
        "cs_bulletin_org_name": "Scottish Government (excl. agencies)",
        "parent_dept": "Scottish Government",
    },
    "serious-fraud-office": {
        "dept_key": "SFO",
        "grid_name": "Serious Fraud Office",
        "cs_bulletin_org_name": "Serious Fraud Office",
        "parent_dept": "Attorney General's Departments",
    },
    "social-security-scotland": {
        "dept_key": "SSS",
        "grid_name": "Social Security Scotland",
        "cs_bulletin_org_name": "Social Security Scotland",
        "parent_dept": "Scottish Government",
    },
    "the-national-archives": {
        "dept_key": "TNA",
        "grid_name": "The National Archives",
        "cs_bulletin_org_name": "The National Archives",
        "parent_dept": "The National Archives",
    },
    "the-supreme-court-of-the-united-kingdom": {
        "dept_key": "UK Supreme Court",
        "grid_name": "The Supreme Court of the United Kingdom",
        "cs_bulletin_org_name": "UK Supreme Court",
        "parent_dept": "UK Supreme Court",
    },
    "uk-debt-management-office": {
        "dept_key": "DMO",
        "grid_name": "UK Debt Management Office",
        "cs_bulletin_org_name": "UK Debt Management Office",
        "parent_dept": "HM Treasury",
    },
    "uk-export-finance": {
        "dept_key": "UKEF",
        "grid_name": "UK Export Finance",
        "cs_bulletin_org_name": "UK Export Finance",
        "parent_dept": "UK Export Finance",
    },
    "uk-health-security-agency": {
        "dept_key": "UKHSA",
        "grid_name": "UK Health Security Agency",
        "cs_bulletin_org_name": "UK Health Security Agency",
        "parent_dept": "Department of Health and Social Care",
    },
    "uk-hydrographic-office": {
        "dept_key": "UKHO",
        "grid_name": "UK Hydrographic Office",
        "cs_bulletin_org_name": "UK Hydrographic Office",
        "parent_dept": "Ministry of Defence",
    },
    "uk-space-agency": {
        "dept_key": "UK Space Agency",
        "grid_name": "UK Space Agency",
        "cs_bulletin_org_name": "UK Space Agency",
        "parent_dept": "Department for Science, Innovation and Technology",
    },
    "uk-statistics-authority": {
        "dept_key": "UK Stats Authority",
        "grid_name": "UK Statistics Authority",
        "cs_bulletin_org_name": "UK Statistics Authority",
        "parent_dept": "UK Statistics Authority",
    },
    "valuation-office-agency": {
        "dept_key": "VOA",
        "grid_name": "Valuation Office Agency",
        "cs_bulletin_org_name": "Valuation Office Agency",
        "parent_dept": "HM Revenue and Customs",
    },
    "vehicle-certification-agency": {
        "dept_key": "VCA",
        "grid_name": "Vehicle Certification Agency",
        "cs_bulletin_org_name": "Vehicle Certification Agency",
        "parent_dept": "Department for Transport",
    },
    "veterinary-medicines-directorate": {
        "dept_key": "VMD",
        "grid_name": "Veterinary Medicines Directorate",
        "cs_bulletin_org_name": "Veterinary Medicines Directorate",
        "parent_dept": "Department for Environment, Food and Rural Affairs",
    },
    "welsh-government": {
        "dept_key": "Welsh Government",
        "grid_name": "Welsh Government",
        "cs_bulletin_org_name": "Welsh Government (excl. agencies)",
        "parent_dept": "Welsh Government",
    },
    "welsh-revenue-authority": {
        "dept_key": "WRA",
        "grid_name": "Welsh Revenue Authority",
        "cs_bulletin_org_name": "Welsh Revenue Authority",
        "parent_dept": "Welsh Government",
    },
    "wilton-park": {
        "dept_key": "Wilton Park",
        "grid_name": "Wilton Park",
        "cs_bulletin_org_name": "Wilton Park",
        "parent_dept": "Foreign, Commonwealth and Development Office",
    },
}


# A dictionary to map professions from GRID to the CS statistics counterparts
PROFESSION_MAPPING: dict[str, str] = {
    "Government Property Profession": "Property",
    "Government Finance": "Finance",
    "Government Digital and Data Profession": "Digital and Data",
    "Other": "Other",
    "Government Project Delivery Profession": "Project Delivery",
    "Knowledge & Information Management": "Knowledge and Information Management",
    "Human Resources": "Human Resources",
    "Medical Profession": "Clinical",
    "Government Economic Service": "Economics",
    "Policy Profession": "Policy",
    "Planning Profession": "Planning",
    "Operational Delivery Profession": "Operational Delivery",
    "Internal Audit": "Internal Audit",
    "Government Corporate Finance": "Corporate Finance",
    "Tax Profession": "Tax",
    "Psychology Profession": "Occupational Psychology",
    "Government Science and Engineering": "Science and Engineering",
    "Government Communication Service": "Communications",
    "Government Legal Service": "Legal",
    "Counter Fraud Professions": "Counter Fraud",
    "Security Profession": "Security",
    "Government Operational Research Service": "Operational Research",
    "Intelligence Analysis": "Intelligence Analysis",
    "Government Statistical Service": "Statistics",
    "International Trade Profession": "Other",  # Not found in the UKCS stastistics
    "Government Social Research": "Social Research",
    "Government Veterinary Profession": "Veterinarian",
    "Planning Inspectors": "Planning Inspectors",
    "Commercial": "Commercial",
    "nan": "Unknown",
    "Government Legal Profession": "Legal",
}


# A dictionary to map grade strings in GRID to the CS statistics counterparts
GRADE_MAPPING = {
    # Maps raw GRID grades to CS Statistics grades.
    # General Rule: If the GRID grade spans two grades, assign to the lower one.
    # If it spans more than that, we should flag and remove the observation.
    # Doesn't apply for AA/AO because not many departments have many AAs, so in
    # reality these are likely AOs.
    # SCS values
    "SCS Pay Band 1": "SCS level",
    "SCS Pay Band 2": "SCS level",
    "SCS Pay Band 3": "SCS level",
    "SCS Pay Band 4": "SCS level",
    "SCS Pay Band 1,SCS Pay Band 2,SCS Pay Band 3,SCS Pay Band 4": "SCS level",
    "SCS Pay Band 1,Other": "SCS level",
    "SCS Pay Band 1,SCS Pay Band 2,SCS Pay Band 3": "SCS level",
    "SCS Pay Band 1,SCS Pay Band 2": "SCS level",
    "SCS Pay Band 2,SCS Pay Band 3": "SCS level",
    "SCS Pay Band 2,Other": "SCS level",
    "SCS Pay Band 2,SCS Pay Band 3,SCS Pay Band 4": "SCS level",
    "SCS Pay Band 3,Industrial": "SCS level",
    # Grade 6
    "Grade 6": "G6",
    "Grade 6,Other": "G6",
    "Grade 6,SCS Pay Band 1": "G6",
    "Grade 6,SCS Pay Band 1,SCS Pay Band 2,SCS Pay Band 3,SCS Pay Band 4": "G6",
    "Grade 6,SCS Pay Band 1,Other": "G6",
    # Grade 7
    "Grade 7": "G7",
    "Grade 7,Grade 6": "G7",
    "Grade 7,Other": "G7",
    "Grade 7,Grade 6,Other": "G7",
    # Senior Executive Officer
    "Senior Executive Officer": "SEO",
    "Senior Executive Officer,Grade 7": "SEO",
    "Senior Executive Officer,Other": "SEO",
    "Senior Executive Officer,Grade 7,Other": "SEO",
    # Higher Executive Officer
    "Higher Executive Officer": "HEO",
    "Higher Executive Officer,Senior Executive Officer": "HEO",
    "Higher Executive Officer,Other": "HEO",
    "Higher Executive Officer,Senior Executive Officer,Other": "HEO",
    # Executive Officer
    "Executive Officer": "EO",
    "Executive Officer,Higher Executive Officer": "EO",
    "Executive Officer,Other": "EO",
    "Executive Officer,Higher Executive Officer,Other": "EO",
    "Executive Officer,Industrial": "EO",
    # Administrative Officer
    "Administrative Officer": "AO",
    "Administrative Officer,Executive Officer": "AO",
    "Administrative Officer,Other": "AO",
    "Administrative Officer,Executive Officer,Other": "AO",
    "Administrative Officer,Industrial": "AO",
    "Administrative Assistant,Administrative Officer,Other": "AO",
    "Administrative Assistant,Administrative Officer": "AO",
    # Administrative Assistant
    "Administrative Assistant": "AA",
    "Administrative Assistant,Other": "AA",
    # Other/Industrial -> keep as is: assuming they're AO massively distorts small departments
    "Other": "Other",
    "Industrial": "Industrial",
    "Industrial,Other": "Industrial",
    # Invalid grade values
    "Grade 7,Grade 6,SCS Pay Band 1": "invalid",
    "Senior Executive Officer,Grade 6": "invalid",
    "Senior Executive Officer,Grade 7,Grade 6": "invalid",
    "Senior Executive Officer,Grade 7,SCS Pay Band 1": "invalid",
    "Higher Executive Officer,Senior Executive Officer,Grade 7": "invalid",
    "Higher Executive Officer,Senior Executive Officer,Grade 7,Grade 6": "invalid",
    "Higher Executive Officer,Senior Executive Officer,Grade 7,Grade 6,SCS Pay Band 1": "invalid",
    "Higher Executive Officer,Grade 7": "invalid",
    "Higher Executive Officer,Senior Executive Officer,Grade 6": "invalid",
    "Higher Executive Officer,Grade 7,Grade 6": "invalid",
    "Higher Executive Officer,Senior Executive Officer,Grade 7,Grade 6,SCS Pay Band 1,SCS Pay Band 2": "invalid",
    "Executive Officer,Higher Executive Officer,Senior Executive Officer,Grade 7,Grade 6": "invalid",
    "Executive Officer,Higher Executive Officer,Senior Executive Officer": "invalid",
    "Executive Officer,Higher Executive Officer,Senior Executive Officer,Grade 7": "invalid",
    "Executive Officer,Senior Executive Officer": "invalid",
    "Executive Officer,Higher Executive Officer,Grade 7": "invalid",
    "Executive Officer,Grade 7": "invalid",
    "Executive Officer,Higher Executive Officer,Grade 7,Other": "invalid",
    "Executive Officer,Higher Executive Officer,Senior Executive Officer,Grade 7,Grade 6,Other": "invalid",
    "Executive Officer,Senior Executive Officer,Grade 7,Grade 6": "invalid",
    "Executive Officer,Higher Executive Officer,Senior Executive Officer,Grade 7,Grade 6,SCS Pay Band 1": "invalid",
    "Administrative Officer,Executive Officer,Higher Executive Officer,Senior Executive Officer,Grade 7,Grade 6": "invalid",
    "Administrative Assistant,Administrative Officer,Executive Officer,Higher Executive Officer,Senior Executive Officer,Grade 7,Grade 6": "invalid",
    "Administrative Officer,Executive Officer,Higher Executive Officer": "invalid",
    "Administrative Officer,Executive Officer,Higher Executive Officer,Senior Executive Officer": "invalid",
    "Administrative Officer,Higher Executive Officer": "invalid",
    "Administrative Officer,Higher Executive Officer,Grade 7": "invalid",
    "Administrative Officer,Senior Executive Officer": "invalid",
    "Administrative Officer,Higher Executive Officer,Senior Executive Officer,Grade 7": "invalid",
    "Administrative Officer,Executive Officer,Senior Executive Officer,Grade 7,Grade 6": "invalid",
    "Administrative Assistant,Administrative Officer,Executive Officer": "invalid",
    "Administrative Assistant,Administrative Officer,Executive Officer,Higher Executive Officer,Senior Executive Officer": "invalid",
    "Administrative Assistant,Higher Executive Officer": "invalid",
    "Administrative Assistant,Executive Officer": "invalid",
    "Administrative Assistant,Grade 7": "invalid",
    "Administrative Assistant,Administrative Officer,Executive Officer,Other": "invalid",
    "Administrative Assistant,Senior Executive Officer": "invalid",
    "Administrative Assistant,Administrative Officer,Executive Officer,Higher Executive Officer,Senior Executive Officer,Grade 7": "invalid",
    "Administrative Assistant,Executive Officer,Higher Executive Officer,Senior Executive Officer,Grade 7,Grade 6": "invalid",
    "Administrative Assistant,Administrative Officer,Executive Officer,Higher Executive Officer,Grade 7,Grade 6": "invalid",
    "Administrative Officer,Executive Officer,Higher Executive Officer,Senior Executive Officer,Grade 7,Grade 6,SCS Pay Band 1,SCS Pay Band 2,SCS Pay Band 3,SCS Pay Band 4": "invalid",
}


CS_STATS_BROWSER_URLS = {
    "organisation": {
        "filename": "organisations.csv",
        "url": "https://civil-service-statistics.jdac.service.cabinetoffice.gov.uk/data_browser_2026/Organisation/data.csv",
    },
    "grade": {
        "filename": "grades.csv",
        "url": "https://civil-service-statistics.jdac.service.cabinetoffice.gov.uk/data_browser_2026/Responsibility_level_ungrouped/data.csv",
    },
    "profession": {
        "filename": "professions.csv",
        "url": "https://civil-service-statistics.jdac.service.cabinetoffice.gov.uk/data_browser_2026/Profession_of_post/data.csv",
    },
    "salary": {
        "filename": "salaries.csv",
        "url": "https://civil-service-statistics.jdac.service.cabinetoffice.gov.uk/data_browser_2026/Organisation/Responsibility_level_ungrouped/data.csv",
    },
    "salary_by_dept_grade_prof": {
        "filename": "salaries_by_dept_grade_prof.csv",
        "url": "https://civil-service-statistics.jdac.service.cabinetoffice.gov.uk/data_browser_2026/Organisation/Profession_of_post/Responsibility_level_ungrouped/data.csv",
    },
    "profession_by_dept": {
        "filename": "profession_by_dept.csv",
        "url": "https://civil-service-statistics.jdac.service.cabinetoffice.gov.uk/data_browser_2026/Organisation/Profession_of_post/data.csv",
    },
}


# Raking
EXTRAPOLATION_THRESHOLD = 12.0  # max FTE per surviving vacancy, department level


COVERAGE_THRESHOLD_3D = 80.0  # min % of departmental FTE weightable under 3D


INVALID_VACANCIES = ["265182"]

# Boiler plate job description we want filtered out from GRID
unwanted_job_descriptions = [
    "Details referring to job description and person specification, as well as salary, pension scheme, closing dates and points of contact details specific to each role can be found on NHS Jobs. To view and apply for this vacancy please see the follow link and apply section below. You will need to register for an account on NHS Jobs to apply. You will also be able to set up notifications based on your preferred location, keeping you fully informed when new vacancies are advertised.",
    "This role is offered by the Foreign, Commonwealth and Development Office (FCDO). Any successful candidate who was an FCO or DFID employee on 31 August 2020 will initially remain on their legacy Department's Terms and Conditions.",
    "Details referring to job description and person specification, as well as salary, pension scheme, closing dates and points of contact details specific to each role can be found on NHS Jobs. To view and apply for this vacancy please see the link and apply section below. You will need to register for an account on NHS Jobs to apply. You will also be able to set up notifications based on your preferred location, keeping you fully informed when new vacancies are advertised.",
    "Details referring to job description and person specification, as well as salary, pension scheme, closing dates and points of contact details specific to each role can be found on NHS Jobs. To view and apply for this vacancy please see the following link and apply section below. You will need to register for an account on NHS Jobs to apply. You will also be able to set up notifications based on your preferred location, keeping you fully informed when new vacancies are advertised.",
    "The FCDO pursues our national interests and projects the UK as a force for good in the world. We promote the interests of British citizens, safeguard the UK's security, defend our values, reduce poverty and tackle global challenges with our international partners. Take from Job Profile.",
    "The programme consists of a number of structured development activities and self-directed learning, including individual coaching to support self awareness and career planning. Participants will be able to practise a wide range of skills including application writing and interviewing, networking and leadership and critical thinking skills. Beyond Boundaries is a self-managed learning programme, which means that learners take responsibility for their own development. This is not just about what learning is taken away from the programme interventions and applied to day to day work; it's about being proactive in pursuit of own learning objectives and career goals by creating and undertaking learning and development opportunities and personal reflection. Participants will learn in cohorts of 16 peers of a similar grade. These grade groups are AA/AO, EO and HEO/SEO. Further details outlining the programme activities can be found in the Candidate Information Pack.",
    "The programme consists of a number of structured development activities and self-directed learning, including individual coaching to support self awareness and career planning. Participants will be able to practise a wide range of skills including application writing and interviewing, networking and leadership and critical thinking skills. Further details outlining the programme activities can be found in the Candidate Information Pack. Participants will learn in cohorts of 16 peers of a similar grade. The programme is intended to give you some of the skills, motivation and confidence that will help you articulate your strengths to apply for new opportunities and to reach your full potential. It will also help you make sense of the abundance of opportunities that exist, and the processes that enable your professional and career development. Although there are a number of mandatory learning interventions, you will be expected to take responsibility for your own learning during your time on the programme. This involves ensuring you are available to attend all of the events and flagging where this might be an issue at the earliest opportunity. You will also be expected to actively participate in opportunities to help to provide stretch in your specific areas of development.",
    "The Foreign & Commonwealth Office (FCO) and the Department for International Development (DfID) merged on 2 September 2020 to form the Foreign, Commonwealth & Development Office (FCDO). The FCDO pursues our national interests and projects the UK as a force for good in the world. We promote the interests of British citizens, safeguard the UK's security, defend our values, reduce poverty and tackle global challenges with our international partners. Any successful candidate who was an FCO or DFID employee on 31 August 2020 will be appointed on their current Terms and Conditions. Where the successful candidate is appointed from elsewhere in the Civil Service or externally, they will join the organisation on FCDO Terms and Conditions.",
    "As the independent regulator of general and vocational qualifications in England, Ofqual's job is to maintain standards, to make sure that qualifications accurately reflect the knowledge and skills people need to achieve their goals. Over the last nine years, general qualifications in England - GCSEs, AS and A levels - have been overhauled to ensure that they compare with the best in the world. Technical qualifications and apprenticeships are now undergoing a similar programme of reform, with the introduction of the new T Levels in 2020 at the forefront of this.",
    "As England's independent regulator of qualifications, examinations and assessments Ofqual's job is to maintain standards, to make sure that qualifications accurately reflect the knowledge and skills people have attained through a range of different learning methods. We work with other Civil Service departments across the spectrum of education to deliver on our priorities and to ensure the best possible outcomes for learners in England",
    'Subject to the successful candidate satisfying the "general qualification" provision within the meaning of section 71 of the Courts and Service Act 1990, the post is designated by the Director of Public Prosecutions as a Crown Prosecutor under the terms of section 1 (3) of the prosecution of Offences Act (1985).',
    "The FCDO pursues our national interests and projects the UK as a force for good in the world. We promote the interests of British citizens, safeguard the UK's security, defend our values, reduce poverty and tackle global challenges with our international partners. You will be required to remain in this role for the full duration specified in the Job Profile. **Add text from 'Purpose of Job' section on Job Profile**",
    "Please see the attached Vacancy Notice. If you need to contact Defence Business Services (DBS) regarding this vacancy, please indicate that your vacancy is being handled by Resourcing Team 1. If you experience any technical difficulties with the CS Jobs system, please contact the Support Team at candidate@support.oleeo.com",
    "Please note that accreditation is NOT an application for a specific post, a means of getting promoted or a broader civil service recruitment. It is a technical assessment of professional competencies enabling access to a professional community of practice both within FCDO and across Government. As well as providing access to significant Continuing Professional Development opportunities, it is a signifier to recruiters that a candidate meets minimum technical standards.",
    "The Matrix Programme is a transformation initiative involving 9 Departments (listed at end). It is a unique cross-government collaboration that will unlock simpler, better, and more joined up working by bringing together the systems, processes, and ways of working for key government functions such as HR, Finance or Procurement, commonly known as Shared Services. The Programme is part of the Shared Services Team, a fantastic team passionate about improving the existing shared services experience for DSIT and its partners whilst working todeliverthe strategic objectives of the new Government Shared Services Strategy! We will deliver the Government's three overarching objectives for shared services: 1. Better experience for all users - intuitive systems that are easy to use and mobile-enabled. 2. Efficiency and value for money - better systems and services which support productivity and seek to reduce costs. 3. Standardised processes and data - support interoperability, making it easier to understand and compare corporate data. Departments forming the Matrix Programme: Cabinet Office (CO), Department for Science Innovation and Technology (DSIT), Department for Culture Media and Sport (DCMS), Department for Business Trade (DBT), Department of Energy Security and Net Zero (DESNZ), Department for Education (DfE), Department for Health and Social Care (DHSC), Attorney's General Office (AGO) and His Majesty's Treasury (HMT).",
    "Details referring to job description and person specification, as well as salary, pension scheme, closing dates and points of contact details specific to each rile can be found on NHS Jobs. To view and apply for this vacancy please see the following link and apply section below. You will need to register for an account on NHS Jobs to apply. You will also be able to set up notifications based on your preferred location, keeping you fully informed when new vacancies are advertised.",
    "The FCDO pursues our national interests and projects the UK as a force for good in the world. We promote the interests of British citizens, safeguard the UK's security, defend our values, reduce poverty and tackle global challenges with our international partners.",
    'Please see the attached Vacancy Notice. If you need to contact Defence Business Services (DBS) regarding this vacancy, please indicate that your vacancy is being handled by Resourcing Team 1. If you experience any technical difficulties in viewing the CS Jobs system, please contact the Support Team at candidate@support.oleeo.com ". Should you experience difficulties with accessing the OCCAR application form please contact desirg-occar-vacancies@mod.gov.uk .',
    "The Intellectual Property Office (IPO) is the official UK government body responsible for intellectual property (IP) rights including patents, designs, trademarks and copyright. Our ambition is to be the best IP office, and we intend to achieve this by: Delivering excellent IP services Creating a world leading IP environment Making the IPO a brilliant place to work We employ technically excellent people to help us deliver our IP services through cutting edge technologies.",
    "This role is offered by the Foreign, Commonwealth and Development Office (FCDO). Any successful candidate who was an FCO or DFID employee on 31 August 2020 will initially remain on their legacy Department’s Terms and Conditions.",
    "The Foreign & Commonwealth Office (FCO) and the Department for International Development (DfID) merged on 2 September 2020 to form the Foreign, Commonwealth & Development Office (FCDO). The FCDO pursues our national interests and projects the UK as a force for good in the world. We promote the interests of British citizens, safeguard the UK’s security, defend our values, reduce poverty and tackle global challenges with our international partners. Any successful candidate who was an FCO or DFID employee on 31 August 2020 will be appointed on their current Terms and Conditions. Where the successful candidate is appointed from elsewhere in the Civil Service or externally, they will join the organisation on FCDO Terms and Conditions.",
    "As the independent regulator of general and vocational qualifications in England, Ofqual’s job is to maintain standards, to make sure that qualifications accurately reflect the knowledge and skills people need to achieve their goals. Over the last nine years, general qualifications in England – GCSEs, AS and A levels – have been overhauled to ensure that they compare with the best in the world. Technical qualifications and apprenticeships are now undergoing a similar programme of reform, with the introduction of the new T Levels in 2020 at the forefront of this.",
    "As England’s independent regulator of qualifications, examinations and assessments Ofqual’s job is to maintain standards, to make sure that qualifications accurately reflect the knowledge and skills people have attained through a range of different learning methods. We work with other Civil Service departments across the spectrum of education to deliver on our priorities and to ensure the best possible outcomes for learners in England",
    "Subject to the successful candidate satisfying the “general qualification” provision within the meaning of section 71 of the Courts and Service Act 1990, the post is designated by the Director of Public Prosecutions as a Crown Prosecutor under the terms of section 1 (3) of the prosecution of Offences Act (1985).",
    "The Matrix Programme is a transformation initiative involving 9 Departments (listed at end). It is a unique cross-government collaboration that will unlock simpler, better, and more joined up working by bringing together the systems, processes, and ways of working for key government functions such as HR, Finance or Procurement, commonly known as Shared Services. The Programme is part of the Shared Services Team, a fantastic team passionate about improving the existing shared services experience for DSIT and its partners whilst working todeliverthe strategic objectives of the new Government Shared Services Strategy! We will deliver the Government's three overarching objectives for shared services: 1. Better experience for all users - intuitive systems that are easy to use and mobile-enabled. 2. Efficiency and value for money - better systems and services which support productivity and seek to reduce costs. 3. Standardised processes and data - support interoperability, making it easier to understand and compare corporate data. Departments forming the Matrix Programme: Cabinet Office (CO), Department for Science Innovation and Technology (DSIT), Department for Culture Media and Sport (DCMS), Department for Business Trade (DBT), Department of Energy Security and Net Zero (DESNZ), Department for Education (DfE), Department for Health and Social Care (DHSC), Attorney’s General Office (AGO) and His Majesty's Treasury (HMT).",
    "Please see the attached Vacancy Notice. If you need to contact Defence Business Services (DBS) regarding this vacancy, please indicate that your vacancy is being handled by Resourcing Team 1. If you experience any technical difficulties in viewing the CS Jobs system, please contact the Support Team at candidate@support.oleeo.com ”. Should you experience difficulties with accessing the OCCAR application form please contact desirg-occar-vacancies@mod.gov.uk .",
    "The Intellectual Property Office (IPO) is the official UK government body responsible for intellectual property (IP) rights including patents, designs, trademarks and copyright. Our ambition is to be the best IP office, and we intend to achieve this by: • Delivering excellent IP services • Creating a world leading IP environment • Making the IPO a brilliant place to work We employ technically excellent people to help us deliver our IP services through cutting edge technologies.",
    "The programme consists of a number of structured development activities and self-directed learning, including individual coaching to support self awareness and career planning. Participants will be able to practise a wide range of skills including application writing and interviewing, networking and leadership and critical thinking skills. Beyond Boundaries is a self-managed learning programme, which means that learners take responsibility for their own development. This is not just about what learning is taken away from the programme interventions and applied to day to day work; it’s about being proactive in pursuit of own learning objectives and career goals by creating and undertaking learning and development opportunities and personal reflection. Participants will learn in cohorts of 16 peers of a similar grade. These grade groups are AA/AO, EO and HEO/SEO. Further details outlining the programme activities can be found in the Candidate Information Pack.",
    "About HSE At the Health and Safety Executive (HSE), we believe everyone has the right to come home safe and well from their job. We use world-leading science to prevent work-related death, injury and ill health. We don’t just inspect and investigate, we also provide advice and training to organisations. All this together means we help keep people safe, boost productivity, support the UK economy and contribute to a fairer society. About ITF Core to all of HSE’s work is the data, information and knowledge that we have gathered and developed, and which is used to support our field force of inspectors, scientists and policy makers. Information Technology plays a vital role in assuring that HSE staff have the right information, in the right place and at the right time. Utilising Mobile Device Management technologies to deploy, build and control smart phones, tablets and laptops, HSE has been a leading government department at the forefront of innovative remote working solutions and is continuously looking at how to better support its 1500 inspectors and visiting officers in the field. Newly refreshed server and data storage solutions, are securely hosted in modern commercial data centres and monitored by HSE’s Security Operations Centre. Using industry-leading components, management tools and software, the data centres have undergone significant investment in recent years to ensure that services are reliable and robust. HSE is now developing on a roadmap to replace its core business systems. Major digital services programmes are underway, operating with the guidance of the Government Digital Service’s service manual and TCOP, to redesign how HSE delivers its work and how new technologies can improve operational processes. Collaboration and IoT are integral to HSE’s future, supporting remote inspections and compliance monitoring.",
    "The FCDO pursues our national interests and projects the UK as a force for good in the world. We promote the interests of British citizens, safeguard the UK's security, defend our values, reduce poverty and tackle global challenges with our international partners. **Add text from 'Purpose of Job' section on Job Profile**",
    "Beyond Boundaries is a formal 12-month cross-government ",
]


# --- As at commit e246377: what release v2's GRID cleaning used -----------------------------
_ABOLISHED_BODIES = {
    "institute-for-apprenticeships-and-technical-education": {
        "dept_key": "IfATE",
        "grid_name": "Institute for Apprenticeships and Technical Education",
        "cs_bulletin_org_name": "Institute for Apprenticeships and Technical Education",
        "parent_dept": "Department for Education",
    },
    "national-infrastructure-commission": {
        "dept_key": "NIC",
        "grid_name": "National Infrastructure Commission",
        "cs_bulletin_org_name": "National Infrastructure Commission",
        "parent_dept": "HM Treasury",
    },
}
CLEANING_DEPARTMENTS = {**DEPARTMENTS, **_ABOLISHED_BODIES}
CLEANING_GRADE_MAPPING = {
    **GRADE_MAPPING,
    "Administrative Assistant,Administrative Officer,Executive Officer,Higher Executive Officer,"
    "Senior Executive Officer,Grade 7,Grade 6": "invalid",
}
