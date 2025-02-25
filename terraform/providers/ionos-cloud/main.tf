terraform {

  backend "s3" {
    bucket = "customerintelligence"
    key = "ml_platform/terraform.tfstate"
    region = "eu-central-1"
    endpoints = {
      s3 = "http://s3-de-central.profitbricks.com"
    }
    encrypt = true
    skip_credentials_validation = true
    skip_requesting_account_id = true
  }

  required_providers {
    ionoscloud = {
      source = "ionos-cloud/ionoscloud"
      version = ">= 6.4.10"
    }
  }
}

provider "ionoscloud" {
    # For his authorization to work, 
    # environment variable TF_VAR_ionos_cloud must be available 
    token = "${var.ionos_token}" 
}

resource "ionoscloud_k8s_cluster" "example" {
  name                  = var.cluster_name
  k8s_version           = var.k8s_version
  maintenance_window {
    day_of_the_week     = "Sunday"
    time                = "09:00:00Z"
  }
}

resource "ionoscloud_k8s_node_pool" "example" {
  datacenter_id         = "${var.datacenter_id}"
  k8s_cluster_id        = ionoscloud_k8s_cluster.example.id
  name                  = "k8sNodePool"
  k8s_version           = ionoscloud_k8s_cluster.example.k8s_version
  maintenance_window {
    day_of_the_week     = "Monday"
    time                = "09:00:00Z"
  } 
  cpu_family            = "INTEL_SKYLAKE"
  availability_zone     = "AUTO"
  storage_type          = "SSD"
  node_count            = var.node_count > 3 ? 3 : var.node_count
  cores_count           = 8
  ram_size              = 30720
  storage_size          = 100
}