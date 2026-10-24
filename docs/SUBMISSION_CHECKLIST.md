# Submission checklist audit

Audit date: 24 October 2026. Source: the two submission-form screenshots supplied by the team. The form says every item must appear in the submitted repository README. This document separates local readiness from verified public access.

| Form item | Local status | Evidence / action |
|---|---|---|
| Team details | Ready | Team TwinPulse; Rimzim Kumari (leader), Aksh Agrawal |
| College/incubator information | Ready | CSE, VIT Bhopal; ECE, VIT Vellore in the README |
| Project title | Ready | GlucoTwin |
| Problem statement | Ready | Dedicated README section |
| Healthcare use case | Ready | Research/clinician review and Type 2 focus in README |
| Technical stack | Ready | README and pinned requirements.txt |
| AI/ML model/framework details | Ready | README includes inputs, estimators, parameters, selection and adaptation outcome |
| 15-20 minute unlisted YouTube demo linked in README | Pending, team handles video | Actual video URL required. See VIDEO_HANDOFF.md for the timing discrepancy |
| Open-source license details | Ready | MIT code LICENSE; CGMacros CC BY-NC-SA 4.0 separately attributed |
| Architecture diagram in PDF/PPT | Ready locally | ../output/pdf/GlucoTwin_Reliability_Architecture.pdf, linked in README |
| Presentation with project details and outcomes in PDF/PPT | Ready locally | ../output/presentation/GlucoTwin_TwinPulse_Reliability_Edition.pptx, linked in README |
| All files/links publicly accessible without extra permissions | Verified for current non-video files | Anonymous requests returned HTTP 200 for README, architecture, presentation and every README file link; the video remains pending |
| Public GitHub repository URL entered in form | Pending | https://github.com/rimzim9934/glucotwin-digital-twin; enter this URL in the organizer form |

## Public-access check before submission

1. Publish the clean GlucoTwin repository package. Its README is at the repository root and uses relative links.
2. Open the repository URL while signed out. Confirm it is public.
3. Follow the architecture, presentation and report links from the rendered README. Download them and confirm they open without authentication.
4. Add the actual unlisted YouTube watch URL, and verify it while signed out.
5. Enter the public GitHub URL in the organizer form. Tick every required box only after its evidence is accessible.
6. Rimzim Kumari, the team leader, submits and saves the platform receipt.

## Evidence caveats

The 11.1% result is lower average 30-minute forecast error, not accuracy. Scoring v2 corrects causal warning emission and censoring. The reliability-edition PDF/PPT and README include the current 162/163 count and three censored alerts. The engineering disruption benchmark measures withholding and reduced availability, not improved predictive accuracy. Older architecture/results/video assets describe earlier builds and are superseded by the currently linked files.

The original stated deadline has passed. The screenshot does not confirm an extension. Check the current organizer submission window.

## Earlier public repository check (before upload)

https://github.com/rimzim9934/glucotwin-digital-twin

Anonymous GitHub API checks on 24 October 2026: repository HTTP 200, private=false, size=0. README endpoint HTTP 404. Root contents endpoint HTTP 404. Repository visibility passes, but required file accessibility does not pass because no files are present.

## Publication verification

The project was pushed to public main on 24 October 2026. Anonymous requests verified README, licenses, architecture PDF, project report, presentation PPTX and every README file link with matching SHA256 content. See reports/github_publication_verification.json. The unlisted video/link and final leader submission remain pending.
