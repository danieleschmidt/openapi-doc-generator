"""
Test suite for GitHub Repository Hygiene Automation

Tests defensive security automation functionality without making actual API calls.
"""

import json
import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from openapi_doc_generator.github_hygiene import GitHubAPI, RepoHygieneBot


class TestGitHubAPI:
    """Test GitHub API client functionality."""
    
    def test_init_with_token(self):
        """Test API client initialization with token."""
        api = GitHubAPI("test-token")
        assert api.token == "test-token"
        assert api.base_url == "https://api.github.com"
        assert "token test-token" in api.headers['Authorization']
    
    def test_init_from_environment(self):
        """Test API client initialization from environment."""
        with patch.dict(os.environ, {'GITHUB_TOKEN': 'env-token'}):
            api = GitHubAPI()
            assert api.token == "env-token"
    
    def test_init_no_token_raises_error(self):
        """Test initialization without token raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GitHub token required"):
                GitHubAPI()
    
    @patch('urllib.request.urlopen')
    def test_request_success(self, mock_urlopen):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.read.return_value = b'{"test": "data"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        api = GitHubAPI("test-token")
        result = api._request('GET', '/test')
        
        assert result == {"test": "data"}
    
    @patch('urllib.request.urlopen')
    def test_request_http_error(self, mock_urlopen):
        """Test API request with HTTP error."""
        import urllib.error
        
        error = urllib.error.HTTPError(
            url="test", code=404, msg="Not Found", hdrs={}, fp=None
        )
        mock_urlopen.side_effect = error
        
        api = GitHubAPI("test-token")
        
        with pytest.raises(Exception, match="GitHub API error 404"):
            api._request('GET', '/test')
    
    @patch.object(GitHubAPI, '_request')
    def test_get_user_repos(self, mock_request):
        """Test getting user repositories."""
        mock_request.return_value = [{"name": "test-repo"}]
        
        api = GitHubAPI("test-token")
        repos = api.get_user_repos()
        
        mock_request.assert_called_with('GET', '/user/repos?per_page=100&affiliation=owner')
        assert repos == [{"name": "test-repo"}]
    
    @patch.object(GitHubAPI, '_request')
    def test_update_repo(self, mock_request):
        """Test updating repository metadata."""
        mock_request.return_value = {"updated": True}
        
        api = GitHubAPI("test-token")
        result = api.update_repo("owner", "repo", {"description": "test"})
        
        mock_request.assert_called_with('PATCH', '/repos/owner/repo', {"description": "test"})
        assert result == {"updated": True}
    
    @patch.object(GitHubAPI, '_request')
    def test_update_topics(self, mock_request):
        """Test updating repository topics."""
        mock_request.return_value = {"names": ["security", "automation"]}
        
        api = GitHubAPI("test-token")
        result = api.update_topics("owner", "repo", ["security", "automation"])
        
        mock_request.assert_called_with('PUT', '/repos/owner/repo/topics', {'names': ["security", "automation"]})
        assert result == {"names": ["security", "automation"]}
    
    @patch.object(GitHubAPI, '_request')
    def test_get_repo_contents_exists(self, mock_request):
        """Test getting existing repository file contents."""
        mock_request.return_value = {"content": "dGVzdA==", "sha": "abc123"}
        
        api = GitHubAPI("test-token")
        result = api.get_repo_contents("owner", "repo", "README.md")
        
        mock_request.assert_called_with('GET', '/repos/owner/repo/contents/README.md')
        assert result == {"content": "dGVzdA==", "sha": "abc123"}
    
    @patch.object(GitHubAPI, '_request')
    def test_get_repo_contents_not_found(self, mock_request):
        """Test getting non-existent repository file contents."""
        mock_request.side_effect = Exception("Not found")
        
        api = GitHubAPI("test-token")
        result = api.get_repo_contents("owner", "repo", "MISSING.md")
        
        assert result == {}
    
    @patch.object(GitHubAPI, '_request')
    @patch.object(GitHubAPI, 'get_repo_contents')
    def test_create_file_new(self, mock_get_contents, mock_request):
        """Test creating new file."""
        mock_get_contents.return_value = {}
        mock_request.return_value = {"created": True}
        
        api = GitHubAPI("test-token")
        result = api.create_file("owner", "repo", "test.txt", "content", "Add test file")
        
        assert result == {"created": True}
        mock_request.assert_called_once()
        
        call_args = mock_request.call_args
        assert call_args[0] == ('PUT', '/repos/owner/repo/contents/test.txt')
        assert 'content' in call_args[1]['data']
        assert 'message' in call_args[1]['data']
    
    @patch.object(GitHubAPI, '_request')
    @patch.object(GitHubAPI, 'get_repo_contents')
    def test_create_file_update_existing(self, mock_get_contents, mock_request):
        """Test updating existing file."""
        mock_get_contents.return_value = {"sha": "existing-sha"}
        mock_request.return_value = {"updated": True}
        
        api = GitHubAPI("test-token")
        result = api.create_file("owner", "repo", "test.txt", "new content", "Update test file")
        
        assert result == {"updated": True}
        call_args = mock_request.call_args
        assert 'sha' in call_args[1]['data']
        assert call_args[1]['data']['sha'] == "existing-sha"
    
    @patch.object(GitHubAPI, '_request')
    def test_create_branch(self, mock_request):
        """Test creating new branch."""
        mock_request.side_effect = [
            {"object": {"sha": "main-sha"}},  # get main ref
            {"created": True}  # create branch
        ]
        
        api = GitHubAPI("test-token")
        result = api.create_branch("owner", "repo", "feature-branch")
        
        assert result == {"created": True}
        assert mock_request.call_count == 2
    
    @patch.object(GitHubAPI, '_request')
    def test_create_pull_request(self, mock_request):
        """Test creating pull request."""
        mock_request.return_value = {"html_url": "https://github.com/owner/repo/pull/1"}
        
        api = GitHubAPI("test-token")
        result = api.create_pull_request("owner", "repo", "Test PR", "Description", "feature")
        
        expected_data = {
            'title': "Test PR",
            'body': "Description",
            'head': "feature",
            'base': "main"
        }
        mock_request.assert_called_with('POST', '/repos/owner/repo/pulls', expected_data)
        assert result == {"html_url": "https://github.com/owner/repo/pull/1"}
    
    @patch.object(GitHubAPI, '_request')
    def test_pin_repositories(self, mock_request):
        """Test pinning repositories."""
        mock_request.return_value = {"pinned": True}
        
        api = GitHubAPI("test-token")
        result = api.pin_repositories(["owner/repo1", "owner/repo2"])
        
        mock_request.assert_called_with('PUT', '/user/pinned_repositories', {'repositories': ["owner/repo1", "owner/repo2"]})
        assert result == {"pinned": True}


class TestRepoHygieneBot:
    """Test repository hygiene bot functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_api = Mock(spec=GitHubAPI)
        self.bot = RepoHygieneBot(self.mock_api)
    
    def test_filter_repositories(self):
        """Test repository filtering logic."""
        repos = [
            {"name": "normal-repo", "fork": False, "archived": False, "is_template": False},
            {"name": "forked-repo", "fork": True, "archived": False, "is_template": False},
            {"name": "archived-repo", "fork": False, "archived": True, "is_template": False},
            {"name": "template-repo", "fork": False, "archived": False, "is_template": True},
        ]
        
        filtered = self.bot.filter_repositories(repos)
        
        assert len(filtered) == 1
        assert filtered[0]["name"] == "normal-repo"
    
    def test_update_repo_metadata_missing_description(self):
        """Test updating repository with missing description."""
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"},
            "description": None,
            "homepage": "https://example.com",
            "topics": ["existing-topic"]
        }
        
        self.mock_api.update_repo.return_value = {"updated": True}
        
        result = self.bot.update_repo_metadata(repo)
        
        assert result is True
        self.mock_api.update_repo.assert_called_once()
        call_args = self.mock_api.update_repo.call_args[0]
        assert call_args[0] == "testuser"
        assert call_args[1] == "test-repo"
        assert "description" in call_args[2]
    
    def test_update_repo_metadata_missing_homepage(self):
        """Test updating repository with missing homepage."""
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"},
            "description": "Test description",
            "homepage": None,
            "topics": ["existing-topic"]
        }
        
        self.mock_api.update_repo.return_value = {"updated": True}
        
        result = self.bot.update_repo_metadata(repo)
        
        assert result is True
        self.mock_api.update_repo.assert_called_once()
        call_args = self.mock_api.update_repo.call_args[0]
        assert "homepage" in call_args[2]
        assert call_args[2]["homepage"] == "https://testuser.github.io"
    
    def test_update_repo_metadata_insufficient_topics(self):
        """Test updating repository with insufficient topics."""
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"},
            "description": "Test description",
            "homepage": "https://example.com",
            "topics": ["one", "two"]
        }
        
        self.mock_api.update_topics.return_value = {"names": ["one", "two", "security"]}
        
        result = self.bot.update_repo_metadata(repo)
        
        assert result is True
        self.mock_api.update_topics.assert_called_once()
        call_args = self.mock_api.update_topics.call_args[0]
        assert len(call_args[2]) >= 5  # Should add topics to reach at least 5
    
    def test_update_repo_metadata_no_changes_needed(self):
        """Test repository that doesn't need metadata updates."""
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"},
            "description": "Complete description",
            "homepage": "https://example.com",
            "topics": ["one", "two", "three", "four", "five", "six"]
        }
        
        result = self.bot.update_repo_metadata(repo)
        
        assert result is False
        self.mock_api.update_repo.assert_not_called()
        self.mock_api.update_topics.assert_not_called()
    
    def test_create_community_files(self):
        """Test creating community files."""
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"}
        }
        
        self.mock_api.create_branch.return_value = {"created": True}
        self.mock_api.get_repo_contents.return_value = {}  # File doesn't exist
        self.mock_api.create_file.return_value = {"created": True}
        
        result = self.bot.create_community_files(repo)
        
        assert result is True
        assert "Created community files for test-repo" in self.bot.changes_made
        
        # Should try to create branch first
        self.mock_api.create_branch.assert_called_with("testuser", "test-repo", "repo-hygiene-bot")
        
        # Should create multiple community files
        assert self.mock_api.create_file.call_count >= 5
    
    def test_create_community_files_existing_files(self):
        """Test skipping existing community files."""
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"}
        }
        
        self.mock_api.create_branch.return_value = {"created": True}
        self.mock_api.get_repo_contents.return_value = {"content": "existing"}  # File exists
        
        result = self.bot.create_community_files(repo)
        
        assert result is False
        self.mock_api.create_file.assert_not_called()
    
    def test_setup_security_scanners(self):
        """Test setting up security scanning workflows."""
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"}
        }
        
        self.mock_api.get_repo_contents.return_value = {}  # Files don't exist
        self.mock_api.create_file.return_value = {"created": True}
        
        result = self.bot.setup_security_scanners(repo)
        
        assert result is True
        assert "Setup security scanners for test-repo" in self.bot.changes_made
        
        # Should create CodeQL, Dependabot, and Scorecard files
        assert self.mock_api.create_file.call_count >= 3
    
    def test_add_sbom_workflows(self):
        """Test adding SBOM generation workflows."""
        repo = {
            "name": "test-repo", 
            "owner": {"login": "testuser"}
        }
        
        self.mock_api.get_repo_contents.return_value = {}
        self.mock_api.create_file.return_value = {"created": True}
        
        result = self.bot.add_sbom_workflows(repo)
        
        assert result is True
        assert "Added SBOM workflows for test-repo" in self.bot.changes_made
        assert self.mock_api.create_file.call_count >= 2
    
    def test_update_readme_badges(self):
        """Test adding security badges to README."""
        import base64
        
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"}
        }
        
        readme_content = "# Test Repo\n\nDescription here"
        encoded_content = base64.b64encode(readme_content.encode('utf-8')).decode('utf-8')
        
        self.mock_api.get_repo_contents.return_value = {"content": encoded_content}
        self.mock_api.create_file.return_value = {"updated": True}
        
        result = self.bot.update_readme_badges(repo)
        
        assert result is True
        assert "Added badges to README for test-repo" in self.bot.changes_made
        self.mock_api.create_file.assert_called_once()
        
        # Check that badges were added
        call_args = self.mock_api.create_file.call_args[0]
        new_content = call_args[2]
        assert "[![License]" in new_content
        assert "[![CI]" in new_content
        assert "[![Security Rating]" in new_content
    
    def test_update_readme_badges_already_exists(self):
        """Test skipping badge injection when badges already exist."""
        import base64
        
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"}
        }
        
        readme_content = "# Test Repo\n\n[![License](https://img.shields.io/badge/license-MIT-blue.svg)]\n\nDescription"
        encoded_content = base64.b64encode(readme_content.encode('utf-8')).decode('utf-8')
        
        self.mock_api.get_repo_contents.return_value = {"content": encoded_content}
        
        result = self.bot.update_readme_badges(repo)
        
        assert result is False
        self.mock_api.create_file.assert_not_called()
    
    def test_archive_stale_repos(self):
        """Test archiving stale repositories."""
        old_date = (datetime.now() - timedelta(days=450)).isoformat() + "Z"
        
        repo = {
            "name": "Main-Project",
            "owner": {"login": "testuser"},
            "updated_at": old_date
        }
        
        self.mock_api.update_repo.return_value = {"archived": True}
        
        result = self.bot.archive_stale_repos(repo)
        
        assert result is True
        assert "Archived stale repository Main-Project" in self.bot.changes_made
        self.mock_api.update_repo.assert_called_with("testuser", "Main-Project", {"archived": True})
    
    def test_archive_stale_repos_wrong_name(self):
        """Test not archiving repos with wrong name."""
        old_date = (datetime.now() - timedelta(days=450)).isoformat() + "Z"
        
        repo = {
            "name": "Other-Project",
            "owner": {"login": "testuser"},
            "updated_at": old_date
        }
        
        result = self.bot.archive_stale_repos(repo)
        
        assert result is False
        self.mock_api.update_repo.assert_not_called()
    
    def test_archive_stale_repos_not_old_enough(self):
        """Test not archiving recent repositories."""
        recent_date = (datetime.now() - timedelta(days=30)).isoformat() + "Z"
        
        repo = {
            "name": "Main-Project",
            "owner": {"login": "testuser"},
            "updated_at": recent_date
        }
        
        result = self.bot.archive_stale_repos(repo)
        
        assert result is False
        self.mock_api.update_repo.assert_not_called()
    
    def test_ensure_readme_sections(self):
        """Test ensuring required README sections exist."""
        import base64
        
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"}
        }
        
        readme_content = "# Test Repo\n\nBasic description"
        encoded_content = base64.b64encode(readme_content.encode('utf-8')).decode('utf-8')
        
        self.mock_api.get_repo_contents.return_value = {"content": encoded_content}
        self.mock_api.create_file.return_value = {"updated": True}
        
        result = self.bot.ensure_readme_sections(repo)
        
        assert result is True
        assert "Added README sections to test-repo" in self.bot.changes_made
        
        call_args = self.mock_api.create_file.call_args[0]
        new_content = call_args[2]
        assert "## ✨ Why this exists" in new_content
        assert "## ⚡ Quick Start" in new_content
        assert "## 🔍 Key Features" in new_content
    
    def test_ensure_readme_sections_already_complete(self):
        """Test skipping when README sections already exist."""
        import base64
        
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"}
        }
        
        readme_content = """# Test Repo

## ✨ Why this exists
## ⚡ Quick Start
## 🔍 Key Features
## 🗺 Road Map
## 🤝 Contributing
"""
        encoded_content = base64.b64encode(readme_content.encode('utf-8')).decode('utf-8')
        
        self.mock_api.get_repo_contents.return_value = {"content": encoded_content}
        
        result = self.bot.ensure_readme_sections(repo)
        
        assert result is False
        self.mock_api.create_file.assert_not_called()
    
    def test_pin_top_repositories(self):
        """Test pinning top repositories by stars."""
        repos = [
            {"name": "repo1", "owner": {"login": "user"}, "stargazers_count": 100},
            {"name": "repo2", "owner": {"login": "user"}, "stargazers_count": 50},
            {"name": "repo3", "owner": {"login": "user"}, "stargazers_count": 200},
        ]
        
        self.mock_api.pin_repositories.return_value = {"pinned": True}
        
        result = self.bot.pin_top_repositories(repos)
        
        assert result is True
        assert "Updated pinned repositories" in self.bot.changes_made
        
        # Should pin repos sorted by stars (highest first)
        call_args = self.mock_api.pin_repositories.call_args[0][0]
        assert call_args[0] == "user/repo3"  # 200 stars
        assert call_args[1] == "user/repo1"  # 100 stars  
        assert call_args[2] == "user/repo2"  # 50 stars
    
    def test_log_metrics(self):
        """Test logging hygiene metrics."""
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"}
        }
        
        metrics = {
            "description_set": True,
            "topics_count": 5,
            "license_exists": True
        }
        
        self.mock_api.create_file.return_value = {"created": True}
        
        result = self.bot.log_metrics(repo, metrics)
        
        assert result is True
        self.mock_api.create_file.assert_called_once()
        
        call_args = self.mock_api.create_file.call_args[0]
        assert call_args[2] == 'metrics/profile_hygiene.json'
        
        # Content should be valid JSON
        json_content = call_args[3]
        parsed = json.loads(json_content)
        assert parsed == metrics
    
    def test_create_hygiene_pr(self):
        """Test creating hygiene pull request."""
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"}
        }
        
        self.bot.changes_made = ["Updated metadata", "Added security scanners"]
        
        self.mock_api.create_pull_request.return_value = {
            "html_url": "https://github.com/testuser/test-repo/pull/1"
        }
        
        result = self.bot.create_hygiene_pr(repo)
        
        assert result == "https://github.com/testuser/test-repo/pull/1"
        
        call_args = self.mock_api.create_pull_request.call_args[0]
        assert call_args[0] == "testuser"
        assert call_args[1] == "test-repo"
        assert call_args[2] == "✨ repo-hygiene-bot update"
        assert "Updated metadata" in call_args[3]
        assert "Added security scanners" in call_args[3]
    
    def test_create_hygiene_pr_no_changes(self):
        """Test not creating PR when no changes made."""
        repo = {
            "name": "test-repo",
            "owner": {"login": "testuser"}
        }
        
        self.bot.changes_made = []
        
        result = self.bot.create_hygiene_pr(repo)
        
        assert result is None
        self.mock_api.create_pull_request.assert_not_called()
    
    def test_run_hygiene_check_integration(self):
        """Test complete hygiene check integration."""
        repos = [
            {
                "name": "test-repo",
                "owner": {"login": "testuser"},
                "description": None,
                "homepage": None,
                "topics": [],
                "fork": False,
                "archived": False,
                "is_template": False,
                "stargazers_count": 10
            }
        ]
        
        self.mock_api.get_user_repos.return_value = repos
        self.mock_api.update_repo.return_value = {"updated": True}
        self.mock_api.update_topics.return_value = {"updated": True}
        self.mock_api.create_branch.return_value = {"created": True}
        self.mock_api.get_repo_contents.return_value = {}
        self.mock_api.create_file.return_value = {"created": True}
        self.mock_api.create_pull_request.return_value = {
            "html_url": "https://github.com/testuser/test-repo/pull/1"
        }
        self.mock_api.pin_repositories.return_value = {"pinned": True}
        
        result = self.bot.run_hygiene_check()
        
        assert result['total_repos'] == 1
        assert result['updated_repos'] == 1
        assert len(result['pull_requests']) == 1
        assert result['pull_requests'][0]['repo'] == 'test-repo'
        assert result['pull_requests'][0]['url'] == 'https://github.com/testuser/test-repo/pull/1'


class TestCommunityFileTemplates:
    """Test community file template generation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_api = Mock(spec=GitHubAPI)
        self.bot = RepoHygieneBot(self.mock_api)
    
    def test_apache_license_template(self):
        """Test Apache license template."""
        license_text = self.bot._get_apache_license()
        
        assert "Apache License" in license_text
        assert "Version 2.0" in license_text
        assert "http://www.apache.org/licenses/" in license_text
        assert len(license_text) > 1000  # Should be substantial content
    
    def test_code_of_conduct_template(self):
        """Test code of conduct template."""
        coc_text = self.bot._get_code_of_conduct()
        
        assert "Contributor Covenant" in coc_text
        assert "Our Pledge" in coc_text
        assert "Our Standards" in coc_text
        assert "harassment-free" in coc_text
    
    def test_contributing_guide_template(self):
        """Test contributing guide template."""
        contributing_text = self.bot._get_contributing_guide()
        
        assert "Contributing" in contributing_text
        assert "Development Setup" in contributing_text
        assert "Conventional Commits" in contributing_text
        assert "pytest" in contributing_text
    
    def test_security_policy_template(self):
        """Test security policy template."""
        security_text = self.bot._get_security_policy()
        
        assert "Security Policy" in security_text
        assert "Reporting a Vulnerability" in security_text
        assert "90 days" in security_text
        assert "CodeQL" in security_text
        assert "Dependabot" in security_text
    
    def test_bug_template_yaml(self):
        """Test bug report template YAML."""
        bug_template = self.bot._get_bug_template()
        
        assert "name: Bug Report" in bug_template
        assert "what-happened" in bug_template
        assert "reproduce" in bug_template
        assert "expected" in bug_template
    
    def test_feature_template_yaml(self):
        """Test feature request template YAML."""
        feature_template = self.bot._get_feature_template()
        
        assert "name: Feature Request" in feature_template
        assert "problem" in feature_template
        assert "solution" in feature_template
        assert "alternatives" in feature_template


class TestWorkflowTemplates:
    """Test GitHub Actions workflow templates."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_api = Mock(spec=GitHubAPI)
        self.bot = RepoHygieneBot(self.mock_api)
    
    def test_codeql_workflow(self):
        """Test CodeQL workflow template."""
        workflow = self.bot._get_codeql_workflow()
        
        assert 'name: "CodeQL"' in workflow
        assert 'github/codeql-action/init@v3' in workflow
        assert 'github/codeql-action/autobuild@v3' in workflow
        assert 'security-events: write' in workflow
    
    def test_dependabot_config(self):
        """Test Dependabot configuration."""
        config = self.bot._get_dependabot_config()
        
        assert 'version: 2' in config
        assert 'package-ecosystem: "pip"' in config
        assert 'package-ecosystem: "github-actions"' in config
        assert 'interval: "weekly"' in config
    
    def test_scorecard_workflow(self):
        """Test OpenSSF Scorecard workflow."""
        workflow = self.bot._get_scorecard_workflow()
        
        assert 'name: Scorecard supply-chain security' in workflow
        assert 'ossf/scorecard-action@v2.3.1' in workflow
        assert 'github/codeql-action/upload-sarif@v3' in workflow
        assert 'security-events: write' in workflow
    
    def test_sbom_workflow(self):
        """Test SBOM generation workflow."""
        workflow = self.bot._get_sbom_workflow()
        
        assert 'name: SBOM Generation' in workflow
        assert 'cyclonedx/github-action@v1' in workflow
        assert 'docs/sbom/latest.json' in workflow
        assert 'contents: write' in workflow
    
    def test_sbom_diff_workflow(self):
        """Test SBOM security diff workflow."""
        workflow = self.bot._get_sbom_diff_workflow()
        
        assert 'name: SBOM Security Diff' in workflow
        assert '@cyclonedx/cyclonedx-cli' in workflow
        assert 'cyclonedx diff' in workflow
        assert 'CRITICAL' in workflow


class TestMainFunction:
    """Test main function and CLI behavior."""
    
    @patch('sys.argv', ['github_hygiene.py', '--help'])
    @patch('builtins.print')
    def test_main_help(self, mock_print):
        """Test main function help output."""
        from openapi_doc_generator.github_hygiene import main
        
        main()
        
        # Should print help information
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        help_text = '\n'.join(print_calls)
        assert "GitHub Repository Hygiene Bot" in help_text
        assert "GITHUB_TOKEN" in help_text
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('sys.argv', ['github_hygiene.py'])
    @patch('builtins.print')
    @patch('sys.exit')
    def test_main_no_token(self, mock_exit, mock_print):
        """Test main function without GitHub token."""
        from openapi_doc_generator.github_hygiene import main
        
        main()
        
        mock_exit.assert_called_with(1)
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        error_text = '\n'.join(print_calls)
        assert "Error:" in error_text
    
    @patch.dict(os.environ, {'GITHUB_TOKEN': 'test-token'})
    @patch('sys.argv', ['github_hygiene.py'])
    @patch('openapi_doc_generator.github_hygiene.RepoHygieneBot')
    @patch('openapi_doc_generator.github_hygiene.GitHubAPI')
    @patch('builtins.print')
    def test_main_success(self, mock_print, mock_api_class, mock_bot_class):
        """Test successful main function execution."""
        from openapi_doc_generator.github_hygiene import main
        
        # Mock the bot and its results
        mock_bot = Mock()
        mock_bot.run_hygiene_check.return_value = {
            'total_repos': 5,
            'updated_repos': 3,
            'pull_requests': [
                {'repo': 'test-repo', 'url': 'https://github.com/user/test-repo/pull/1'}
            ]
        }
        mock_bot_class.return_value = mock_bot
        
        # Mock file writing
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            main()
        
        # Should create API and bot instances
        mock_api_class.assert_called_once()
        mock_bot_class.assert_called_once()
        
        # Should run hygiene check
        mock_bot.run_hygiene_check.assert_called_once()
        
        # Should print results
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        output_text = '\n'.join(print_calls)
        assert "Starting repository hygiene check" in output_text
        assert "Hygiene check complete" in output_text
        assert "Total repositories: 5" in output_text
        
        # Should write results file
        mock_open.assert_called_with('hygiene_results.json', 'w')


if __name__ == '__main__':
    pytest.main([__file__])