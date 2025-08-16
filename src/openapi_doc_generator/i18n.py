"""Internationalization support for OpenAPI Doc Generator.

This module provides comprehensive i18n support for global deployment including:
- Multi-language documentation generation
- Localized error messages and CLI output
- Region-specific formatting and conventions
- GDPR, CCPA, PDPA compliance support
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    KOREAN = "ko"


class ComplianceRegion(Enum):
    """Data protection compliance regions."""
    GDPR = auto()      # European Union
    CCPA = auto()      # California
    PDPA_SINGAPORE = auto()  # Singapore
    PDPA_THAILAND = auto()   # Thailand
    LGPD = auto()      # Brazil
    PIPEDA = auto()    # Canada


@dataclass
class LocalizationConfig:
    """Configuration for localization and compliance."""
    language: SupportedLanguage = SupportedLanguage.ENGLISH
    region: str = "US"
    timezone: str = "UTC"
    currency: str = "USD"
    date_format: str = "YYYY-MM-DD"
    compliance_regions: List[ComplianceRegion] = None

    def __post_init__(self):
        if self.compliance_regions is None:
            self.compliance_regions = []


class InternationalizationManager:
    """Manages internationalization and localization for the application."""

    def __init__(self, config: Optional[LocalizationConfig] = None):
        self.config = config or LocalizationConfig()
        self.translations: Dict[str, Dict[str, str]] = {}
        self.compliance_messages: Dict[ComplianceRegion, Dict[str, str]] = {}
        self._load_translations()
        self._load_compliance_messages()

    def _load_translations(self) -> None:
        """Load translation strings for supported languages."""
        # Base translations - in production would load from external files
        base_translations = {
            "en": {
                "cli.generating_docs": "Generating documentation...",
                "cli.analysis_complete": "Documentation analysis complete",
                "cli.error.file_not_found": "File not found: {file_path}",
                "cli.error.invalid_format": "Invalid output format: {format}",
                "cli.success.docs_generated": "Documentation generated successfully",
                "docs.api_title": "API Documentation",
                "docs.endpoints": "Endpoints",
                "docs.schemas": "Data Models",
                "docs.authentication": "Authentication",
                "docs.errors": "Error Responses",
                "validation.required_field": "This field is required",
                "validation.invalid_type": "Invalid data type",
                "performance.optimization_enabled": "Performance optimization enabled",
                "security.scan_complete": "Security scan completed",
            },
            "es": {
                "cli.generating_docs": "Generando documentación...",
                "cli.analysis_complete": "Análisis de documentación completo",
                "cli.error.file_not_found": "Archivo no encontrado: {file_path}",
                "cli.error.invalid_format": "Formato de salida inválido: {format}",
                "cli.success.docs_generated": "Documentación generada exitosamente",
                "docs.api_title": "Documentación de API",
                "docs.endpoints": "Puntos de Acceso",
                "docs.schemas": "Modelos de Datos",
                "docs.authentication": "Autenticación",
                "docs.errors": "Respuestas de Error",
                "validation.required_field": "Este campo es requerido",
                "validation.invalid_type": "Tipo de dato inválido",
                "performance.optimization_enabled": "Optimización de rendimiento habilitada",
                "security.scan_complete": "Escaneo de seguridad completado",
            },
            "fr": {
                "cli.generating_docs": "Génération de documentation...",
                "cli.analysis_complete": "Analyse de documentation terminée",
                "cli.error.file_not_found": "Fichier introuvable: {file_path}",
                "cli.error.invalid_format": "Format de sortie invalide: {format}",
                "cli.success.docs_generated": "Documentation générée avec succès",
                "docs.api_title": "Documentation API",
                "docs.endpoints": "Points de Terminaison",
                "docs.schemas": "Modèles de Données",
                "docs.authentication": "Authentification",
                "docs.errors": "Réponses d'Erreur",
                "validation.required_field": "Ce champ est requis",
                "validation.invalid_type": "Type de données invalide",
                "performance.optimization_enabled": "Optimisation des performances activée",
                "security.scan_complete": "Analyse de sécurité terminée",
            },
            "de": {
                "cli.generating_docs": "Dokumentation wird erstellt...",
                "cli.analysis_complete": "Dokumentationsanalyse abgeschlossen",
                "cli.error.file_not_found": "Datei nicht gefunden: {file_path}",
                "cli.error.invalid_format": "Ungültiges Ausgabeformat: {format}",
                "cli.success.docs_generated": "Dokumentation erfolgreich erstellt",
                "docs.api_title": "API-Dokumentation",
                "docs.endpoints": "Endpunkte",
                "docs.schemas": "Datenmodelle",
                "docs.authentication": "Authentifizierung",
                "docs.errors": "Fehlerantworten",
                "validation.required_field": "Dieses Feld ist erforderlich",
                "validation.invalid_type": "Ungültiger Datentyp",
                "performance.optimization_enabled": "Leistungsoptimierung aktiviert",
                "security.scan_complete": "Sicherheitsscan abgeschlossen",
            },
            "ja": {
                "cli.generating_docs": "ドキュメントを生成中...",
                "cli.analysis_complete": "ドキュメント解析完了",
                "cli.error.file_not_found": "ファイルが見つかりません: {file_path}",
                "cli.error.invalid_format": "無効な出力形式: {format}",
                "cli.success.docs_generated": "ドキュメントが正常に生成されました",
                "docs.api_title": "API ドキュメント",
                "docs.endpoints": "エンドポイント",
                "docs.schemas": "データモデル",
                "docs.authentication": "認証",
                "docs.errors": "エラーレスポンス",
                "validation.required_field": "この項目は必須です",
                "validation.invalid_type": "無効なデータタイプ",
                "performance.optimization_enabled": "パフォーマンス最適化が有効",
                "security.scan_complete": "セキュリティスキャン完了",
            },
            "zh": {
                "cli.generating_docs": "正在生成文档...",
                "cli.analysis_complete": "文档分析完成",
                "cli.error.file_not_found": "文件未找到: {file_path}",
                "cli.error.invalid_format": "无效的输出格式: {format}",
                "cli.success.docs_generated": "文档生成成功",
                "docs.api_title": "API 文档",
                "docs.endpoints": "端点",
                "docs.schemas": "数据模型",
                "docs.authentication": "身份验证",
                "docs.errors": "错误响应",
                "validation.required_field": "此字段为必填项",
                "validation.invalid_type": "无效的数据类型",
                "performance.optimization_enabled": "性能优化已启用",
                "security.scan_complete": "安全扫描完成",
            }
        }

        self.translations.update(base_translations)
        logger.info(f"Loaded translations for {len(base_translations)} languages")

    def _load_compliance_messages(self) -> None:
        """Load compliance-specific messages and notices."""
        compliance_messages = {
            ComplianceRegion.GDPR: {
                "data_processing_notice": "This tool processes personal data in accordance with GDPR Article 6.",
                "data_retention": "Documentation data is retained for legitimate business purposes only.",
                "user_rights": "You have the right to access, rectify, and delete your personal data.",
                "contact_dpo": "Contact our Data Protection Officer at privacy@company.com",
            },
            ComplianceRegion.CCPA: {
                "data_processing_notice": "This tool may collect personal information as defined by CCPA.",
                "data_retention": "Personal information is retained only as necessary for business purposes.",
                "user_rights": "California residents have rights to know, delete, and opt-out of data sales.",
                "contact_privacy": "Contact privacy@company.com for privacy inquiries.",
            },
            ComplianceRegion.PDPA_SINGAPORE: {
                "data_processing_notice": "Personal data is processed in accordance with Singapore PDPA.",
                "data_retention": "Personal data is retained according to PDPA retention requirements.",
                "user_rights": "You have rights to access and correct your personal data.",
                "contact_dpo": "Contact our Data Protection Officer for privacy matters.",
            },
            ComplianceRegion.LGPD: {
                "data_processing_notice": "Dados pessoais processados de acordo com a LGPD brasileira.",
                "data_retention": "Dados pessoais são mantidos apenas pelo tempo necessário.",
                "user_rights": "Você tem direitos de acesso, correção e exclusão de dados pessoais.",
                "contact_dpo": "Contate nosso Encarregado de Dados em privacidade@empresa.com",
            }
        }

        self.compliance_messages.update(compliance_messages)
        logger.info(f"Loaded compliance messages for {len(compliance_messages)} regions")

    def get_text(self, key: str, language: Optional[SupportedLanguage] = None, **kwargs) -> str:
        """Get localized text for a given key."""
        lang = (language or self.config.language).value

        if lang not in self.translations:
            lang = SupportedLanguage.ENGLISH.value

        text = self.translations[lang].get(key, key)

        # Apply string formatting if kwargs provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to format localized text '{key}': {e}")

        return text

    def get_compliance_notice(self, region: ComplianceRegion, notice_type: str) -> str:
        """Get compliance-specific notice text."""
        if region not in self.compliance_messages:
            return ""

        return self.compliance_messages[region].get(notice_type, "")

    def localize_documentation(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Localize documentation content based on current language settings."""
        localized_doc = doc_content.copy()

        # Localize OpenAPI info section
        if "info" in localized_doc:
            info = localized_doc["info"]
            if "title" not in info or info["title"] == "API":
                info["title"] = self.get_text("docs.api_title")

        # Add localized descriptions to paths
        if "paths" in localized_doc:
            for path, path_info in localized_doc["paths"].items():
                for method, operation in path_info.items():
                    if isinstance(operation, dict):
                        # Add localized summaries if missing
                        if "summary" not in operation or not operation["summary"]:
                            operation["summary"] = f"{method.upper()} {path}"

        # Add compliance notices if required
        for region in self.config.compliance_regions:
            compliance_notice = self.get_compliance_notice(region, "data_processing_notice")
            if compliance_notice:
                if "info" not in localized_doc:
                    localized_doc["info"] = {}
                if "description" not in localized_doc["info"]:
                    localized_doc["info"]["description"] = ""

                localized_doc["info"]["description"] += f"\n\n**Compliance Notice**: {compliance_notice}"

        return localized_doc

    def format_date(self, date_obj) -> str:
        """Format date according to regional preferences."""
        # Simplified date formatting - would use proper locale formatting in production
        formats = {
            "US": "%m/%d/%Y",
            "EU": "%d/%m/%Y",
            "ISO": "%Y-%m-%d",
            "JP": "%Y年%m月%d日",
            "CN": "%Y年%m月%d日"
        }

        format_str = formats.get(self.config.region, formats["ISO"])
        return date_obj.strftime(format_str)

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return [lang.value for lang in SupportedLanguage]

    def set_language(self, language: SupportedLanguage) -> None:
        """Change the current language setting."""
        self.config.language = language
        logger.info(f"Language changed to {language.value}")

    def add_compliance_region(self, region: ComplianceRegion) -> None:
        """Add a compliance region to the configuration."""
        if region not in self.config.compliance_regions:
            self.config.compliance_regions.append(region)
            logger.info(f"Added compliance region: {region.name}")

    def get_region_config(self) -> Dict[str, Any]:
        """Get region-specific configuration settings."""
        return {
            "language": self.config.language.value,
            "region": self.config.region,
            "timezone": self.config.timezone,
            "currency": self.config.currency,
            "date_format": self.config.date_format,
            "compliance_regions": [r.name for r in self.config.compliance_regions]
        }


class GlobalDeploymentManager:
    """Manages global deployment considerations and multi-region support."""

    def __init__(self, i18n_manager: InternationalizationManager):
        self.i18n = i18n_manager
        self.deployment_regions: Dict[str, Dict[str, Any]] = {}
        self._initialize_regions()

    def _initialize_regions(self) -> None:
        """Initialize supported deployment regions with their characteristics."""
        regions = {
            "us-east-1": {
                "name": "US East (Virginia)",
                "languages": ["en", "es"],
                "compliance": [ComplianceRegion.CCPA],
                "data_residency": "US",
                "cdn_endpoints": ["https://cdn-us-east.example.com"],
                "api_gateway": "https://api-us-east.example.com"
            },
            "eu-west-1": {
                "name": "EU West (Ireland)",
                "languages": ["en", "fr", "de", "es", "it"],
                "compliance": [ComplianceRegion.GDPR],
                "data_residency": "EU",
                "cdn_endpoints": ["https://cdn-eu-west.example.com"],
                "api_gateway": "https://api-eu-west.example.com"
            },
            "ap-southeast-1": {
                "name": "Asia Pacific (Singapore)",
                "languages": ["en", "zh", "ja"],
                "compliance": [ComplianceRegion.PDPA_SINGAPORE],
                "data_residency": "APAC",
                "cdn_endpoints": ["https://cdn-ap-southeast.example.com"],
                "api_gateway": "https://api-ap-southeast.example.com"
            },
            "ap-northeast-1": {
                "name": "Asia Pacific (Tokyo)",
                "languages": ["ja", "en"],
                "compliance": [],
                "data_residency": "APAC",
                "cdn_endpoints": ["https://cdn-ap-northeast.example.com"],
                "api_gateway": "https://api-ap-northeast.example.com"
            }
        }

        self.deployment_regions.update(regions)
        logger.info(f"Initialized {len(regions)} deployment regions")

    def get_optimal_region(self, user_language: str, user_region: str) -> str:
        """Determine optimal deployment region for a user."""
        # Simple region selection logic
        region_mapping = {
            "US": "us-east-1",
            "CA": "us-east-1",
            "EU": "eu-west-1",
            "GB": "eu-west-1",
            "FR": "eu-west-1",
            "DE": "eu-west-1",
            "SG": "ap-southeast-1",
            "JP": "ap-northeast-1",
            "CN": "ap-southeast-1",
        }

        return region_mapping.get(user_region, "us-east-1")

    def generate_deployment_config(self, target_regions: List[str]) -> Dict[str, Any]:
        """Generate deployment configuration for specified regions."""
        config = {
            "regions": {},
            "global_settings": {
                "load_balancing": "geo_proximity",
                "failover_enabled": True,
                "data_replication": "cross_region",
                "compliance_monitoring": True
            }
        }

        for region in target_regions:
            if region in self.deployment_regions:
                region_info = self.deployment_regions[region]
                config["regions"][region] = {
                    "name": region_info["name"],
                    "languages": region_info["languages"],
                    "compliance_requirements": [c.name for c in region_info["compliance"]],
                    "endpoints": {
                        "api": region_info["api_gateway"],
                        "cdn": region_info["cdn_endpoints"][0]
                    },
                    "data_residency": region_info["data_residency"],
                    "health_check_endpoint": f"{region_info['api_gateway']}/health"
                }

        return config

    def validate_compliance(self, region: str, data_types: List[str]) -> Dict[str, Any]:
        """Validate compliance requirements for data processing in a region."""
        if region not in self.deployment_regions:
            return {"valid": False, "reason": "Unknown region"}

        region_info = self.deployment_regions[region]
        compliance_regions = region_info["compliance"]

        validation_result = {
            "valid": True,
            "region": region,
            "compliance_regions": [c.name for c in compliance_regions],
            "required_measures": [],
            "data_types_allowed": data_types,
            "recommendations": []
        }

        # Check GDPR requirements
        if ComplianceRegion.GDPR in compliance_regions:
            if "personal_data" in data_types:
                validation_result["required_measures"].extend([
                    "lawful_basis_required",
                    "data_protection_impact_assessment",
                    "consent_management",
                    "right_to_be_forgotten_support"
                ])
                validation_result["recommendations"].append(
                    "Implement privacy by design principles"
                )

        # Check CCPA requirements
        if ComplianceRegion.CCPA in compliance_regions:
            if "personal_information" in data_types:
                validation_result["required_measures"].extend([
                    "consumer_rights_support",
                    "opt_out_mechanisms",
                    "data_sale_disclosure"
                ])

        return validation_result


# Global instances
_i18n_manager: Optional[InternationalizationManager] = None
_deployment_manager: Optional[GlobalDeploymentManager] = None


def get_i18n_manager() -> InternationalizationManager:
    """Get the global i18n manager instance."""
    global _i18n_manager
    if _i18n_manager is None:
        # Auto-detect language from environment
        detected_lang = os.getenv("LANG", "en_US").split("_")[0].lower()
        try:
            language = SupportedLanguage(detected_lang)
        except ValueError:
            language = SupportedLanguage.ENGLISH

        config = LocalizationConfig(language=language)
        _i18n_manager = InternationalizationManager(config)

    return _i18n_manager


def get_deployment_manager() -> GlobalDeploymentManager:
    """Get the global deployment manager instance."""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = GlobalDeploymentManager(get_i18n_manager())

    return _deployment_manager


def localize_text(key: str, **kwargs) -> str:
    """Convenience function to get localized text."""
    return get_i18n_manager().get_text(key, **kwargs)


def set_global_language(language: SupportedLanguage) -> None:
    """Set the global language for the application."""
    get_i18n_manager().set_language(language)
