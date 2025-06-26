from typing import Any, TypedDict, cast
from archaeo_super_prompt.models.chunk_selector import select_end_pages, select_incipit
from archaeo_super_prompt.types.pdfchunks import (
    PDFChunkPerInterventionDataset,
)
from archaeo_super_prompt.types.structured_data import ExtractedStructuredDataSeries
from archaeo_super_prompt.utils import flatten_dict
import dspy

from ..debug_log import forward_warning, print_debug_log
from ..target_types import MagohDocumentBuildingData, MagohUniversityData
from ..signatures.arch_dictionnaries import (
    to_magoh_build_data,
    to_magoh_university_data,
)

from ..signatures.arch_extract_type import (
    ArchaeologicalInterventionContext,
    SourceOfInformationInReport,
    TechnicalInformation,
    ArchivalInformation,
)


class ExtractedInterventionData(TypedDict):
    university: MagohUniversityData
    building: MagohDocumentBuildingData


class ExtractDataFromInterventionReport(dspy.Module):
    def __init__(self):
        self.extract_intervention_context_data = dspy.ChainOfThought(
            ArchaeologicalInterventionContext
        )
        self.extract_report_sources = dspy.ChainOfThought(SourceOfInformationInReport)
        self.extract_intervention_technical_achievements = dspy.ChainOfThought(
            TechnicalInformation
        )
        self.extract_archival_metadata = dspy.ChainOfThought(ArchivalInformation)

    def forward(self, document_ocr_scans__df: PDFChunkPerInterventionDataset):
        document_ocr_scans = document_ocr_scans__df.to_readable_context_string()
        document_incipits = select_incipit(document_ocr_scans__df)
        document_end_pages = select_end_pages(document_ocr_scans__df)
        document_incipits_content = document_incipits.to_readable_context_string()
        document_full_contextual_content = (
            document_incipits + document_end_pages
        ).to_readable_context_string()

        CONTEXT = """You are analysing a Italian official documents about an archaeological intervention and you are going to extract in Italian some information as the archivists in archaeology do."""

        ASSURANCE_CONTEXT = """I have mentionned some information as optional as a document can forget to mention it, then try to think if you can figure it out or if you have to answer nothing for these given fields. For the non optional field, you must answer something as the information is directly written in the content I'll give you."""

        print_debug_log("Requesting document sources...")
        document_source_data = cast(
            SourceOfInformationInReport,
            self.extract_report_sources(
                context=CONTEXT + ASSURANCE_CONTEXT,
                documents_contextual_content=document_full_contextual_content,
            ),
        )
        print_debug_log("Requesting archaeological intervention context...")
        intervention_context = cast(
            ArchaeologicalInterventionContext,
            self.extract_intervention_context_data(
                context=CONTEXT + ASSURANCE_CONTEXT,
                documents_contextual_content=document_full_contextual_content,
            ),
        )
        print_debug_log("Requesting archaeological intervention details...")
        techinal_achievements = cast(
            TechnicalInformation,
            self.extract_intervention_technical_achievements(
                context=CONTEXT + ASSURANCE_CONTEXT,
                documents_full_content=document_ocr_scans,
            ),
        )

        print_debug_log("Requesting document archival metadata...")
        report_archival_metadata = cast(
            ArchivalInformation,
            self.extract_archival_metadata(
                context=CONTEXT + ASSURANCE_CONTEXT,
                documents_incipit=document_incipits_content,
            ),
        )

        final_prediction: ExtractedInterventionData = {
            "university": to_magoh_university_data(
                intervention_context, techinal_achievements, document_source_data
            ),
            "building": to_magoh_build_data(
                intervention_context, document_source_data, report_archival_metadata
            ),
        }

        return dspy.Prediction(
            **flatten_dict(cast(dict[str, dict[str, Any]], final_prediction)),
        )

    def forward_and_type(self, document_ocr_scan__df: PDFChunkPerInterventionDataset):
        result: dspy.Prediction
        try:
            result = cast(
                dspy.Prediction, self(document_ocr_scans__df=document_ocr_scan__df)
            )
        except Exception as e:
            forward_warning(e)
            return None
        return cast(ExtractedStructuredDataSeries, result.toDict())
