from typing import TypedDict, cast
import dspy

from ..debug_log import forward_warning, print_debug_log
from ..target_types import MagohDocumentBuildingData, MagohUniversityData
from ..signatures.arch_dictionnaries import (
    to_magoh_build_data,
    to_magoh_university_data,
)
from ..signatures.input import PDFSources

from ..signatures.arch_extract_type import (
    ArchaeologicalReportCutting,
    ArchaeologicalInterventionContext,
    SourceOfInformationInReport,
    TechnicalInformation,
    ArchivalInformation,
)


class ExtractedInterventionData(TypedDict):
    university: MagohUniversityData
    build: MagohDocumentBuildingData


class ExtractDataFromInterventionReport(dspy.Module):
    def __init__(self):
        self.cut_report = dspy.Predict(ArchaeologicalReportCutting)
        self.extract_intervention_context_data = dspy.ChainOfThought(
            ArchaeologicalInterventionContext
        )
        self.extract_report_sources = dspy.ChainOfThought(SourceOfInformationInReport)
        self.extract_intervention_technical_achievements = dspy.ChainOfThought(
            TechnicalInformation
        )
        self.extract_archival_metadata = dspy.ChainOfThought(ArchivalInformation)

    def forward(self, document_ocr_scan: PDFSources):
        CONTEXT = """You are analysing a Italian official documents about an archaeological intervention and you are going to extract in Italian some information as the archivists in archaeology do."""

        ASSURANCE_CONTEXT = """I have mentionned some information as optional as a document can forget to mention it, then try to think if you can figure it out or if you have to answer nothing for these given fields. For the non optional field, you must answer something as the information is directly written in the content I'll give you."""
        print_debug_log("Requesting document cutting...")
        cuts = document_ocr_scan
        cuts = cast(
            ArchaeologicalReportCutting,
            self.cut_report(
                italian_document_ocr_scan=document_ocr_scan, context=CONTEXT
            ),
        )
        print_debug_log("Requesting document sources...")
        document_source_data = cast(
            SourceOfInformationInReport,
            self.extract_report_sources(
                context=CONTEXT + ASSURANCE_CONTEXT, report_incipit=cuts.incipit
            ),
        )
        print_debug_log("Requesting archaeological intervention context...")
        intervention_context = cast(
            ArchaeologicalInterventionContext,
            self.extract_intervention_context_data(
                context=CONTEXT + ASSURANCE_CONTEXT,
                archaeological_report_incipit=cuts.incipit,
                archaeological_report_body=cuts.body,
            ),
        )
        print_debug_log("Requesting archaeological intervention details...")
        techinal_achievements = cast(
            TechnicalInformation,
            self.extract_intervention_technical_achievements(
                context=CONTEXT + ASSURANCE_CONTEXT,
                archaeological_report_body=cuts.body,
            ),
        )

        print_debug_log("Requesting document archival metadata...")
        report_archival_metadata = (
            cast(
                ArchivalInformation,
                self.extract_archival_metadata(
                    context=CONTEXT + ASSURANCE_CONTEXT,
                    report_archive_office_stamp=cuts.archival_stamp,
                ),
            )
            if cuts.archival_stamp is not None
            else None
        )

        final_prediction: ExtractedInterventionData = {
            "university": to_magoh_university_data(
                intervention_context, techinal_achievements, document_source_data
            ),
            "build": to_magoh_build_data(
                intervention_context, document_source_data, report_archival_metadata
            ),
        }
        return dspy.Prediction(**final_prediction)

    def forward_and_type(self, document_ocr_scan: PDFSources):
        result: dspy.Prediction
        try:
            result = cast(dspy.Prediction, self(document_ocr_scan=document_ocr_scan))
        except Exception as e:
            forward_warning(e)
            return None
        return cast(
            ExtractedInterventionData,
            result.toDict(),
        )
