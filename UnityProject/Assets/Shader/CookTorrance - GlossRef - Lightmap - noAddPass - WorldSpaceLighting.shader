Shader "Custom/CookTorrance - GlossyRef -Lightmap - WorldSpaceLighting"
{
    Properties
    {
        _MainTex ("Diffuse (RGB) Alpha (A)", 2D) = "white" {}
        _Roughness ("Roughness", Range(0.01,1)) = 0.5
        _BumpMap ("Normal (Normal)", 2D) = "bump" {}
        //_Beckmann ("Beckmann Lookup (RGB)", 2D) = "gray" {}
        _Fresnel ("Fresnel Value", Range(0.01,1)) = 0.028
        //_Cutoff ("Alpha Cut-Off Threshold", Range(0,1)) = 0.5
        //_GaussConstant("Blinn NDF Value", Float) = 0.2
		//_Cube("Reflection Map", Cube) = "_Skybox" { TexGen CubeReflect }
		//_EnvLightIntensity ("EnvLightIntensity", Range(0,2)) = 0.5

		_LightMapModifier ("LightMapModifier", Range (0.03, 1)) = 0.5
		_WorldSpaceLightDir("WorldSpaceLightDir", Vector) = (1,3,1,0)
		_LightColor("LightColor", Color) = (0.5, 0.5, 0.5, 1)
    }
    SubShader
    {
        Tags { "Queue" = "Geometry" "IgnoreProjector" = "True" }
        LOD 400

        Pass
        {
        	Name "FORWARD"
			Tags { "LightMode" = "ForwardBase" }
        
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
			#pragma target 3.0
            
            #define UNITY_PASS_FORWARDBASE
            #include "UnityCG.cginc"
			#include "Lighting.cginc"
			#include "AutoLight.cginc"

            //#define USE_REAL_LIGHT
            
            // vertex-to-fragment interpolation data
			struct v2f_surf {
			  float4 pos : SV_POSITION;
			  float2 pack0 : TEXCOORD0;
			  fixed3 lightDir : TEXCOORD1;
			  fixed3 vlight : TEXCOORD2;
			  float3 viewDir : TEXCOORD3;
			  float2 lmap : TEXCOORD4;
			  LIGHTING_COORDS(5,6)
			  
		  		half3 TtoW0 : TEXCOORD7;
				half3 TtoW1 : TEXCOORD8;
				half3 TtoW2 : TEXCOORD9;
			};
			
			struct Input
            {
                float2 uv_MainTex;
                half3 TtoW0;
				half3 TtoW1;
				half3 TtoW2;
				//float3 worldRefl;
            };
            
            
            sampler2D _MainTex, _BumpMap;//, _Beckmann;
            
			sampler2D unity_Lightmap;
			sampler2D unity_LightmapInd;
			
            float4 _MainTex_ST;
            float4 unity_LightmapST;
			//samplerCUBE _Cube;
            float _Fresnel,_Roughness;//_GaussConstant; _EnvLightIntensity;
			fixed4 _WorldSpaceLightDir;
			fixed3 _LightColor;
            float _LightMapModifier;
            
             void surf (Input IN, inout SurfaceOutput o)
            {
                fixed4 albedo = tex2D(_MainTex, IN.uv_MainTex);
                o.Albedo = albedo.rgb;
                o.Alpha = albedo.a;
				
				//o.Emission = texCUBE (_Cube, IN.worldRefl).rgb;

				//fixed4 reflcol = texCUBE (_Cube, IN.worldRefl);
				//o.Emission = reflcol.rgb;

                o.Normal = UnpackNormal(tex2D(_BumpMap, IN.uv_MainTex));
                
                o.Normal = half3(dot(IN.TtoW0, o.Normal.xyz), dot(IN.TtoW1, o.Normal.xyz), dot(IN.TtoW2, o.Normal.xyz));
                
                o.Normal = normalize(o.Normal);
            }
 
            inline fixed4 LightingCookTorrance (SurfaceOutput s, fixed3 lightDir, fixed3 viewDir, fixed atten)
            {
                //clip ( s.Alpha - _Cutoff );
 
                viewDir = normalize ( viewDir );
                lightDir = normalize ( lightDir );
                float3 h = normalize ( lightDir + viewDir );
                
                //float3 r = normalize(reflect(-viewDir, s.Normal));
                
                float NdotL_unsat = dot ( s.Normal, lightDir );
                float NdotH_unsat = dot ( s.Normal, h );
                float NdotL = saturate( NdotL_unsat );
                float NdotH = saturate( NdotH_unsat );
                float NdotV = saturate( dot ( s.Normal, viewDir ) );
                float VdotH = saturate( dot ( viewDir, h ) );
                
                
 
                float geo_numerator = 2.0 * NdotH;
                float geo_b = ( geo_numerator * NdotV ) / VdotH;
                float geo_c = ( geo_numerator * NdotL ) / VdotH;
                float geo = min( 1.0f, min( geo_b, geo_c ) );
 
 
 				
 
 				////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //float roughness = tex2D( _Beckmann, float2 ( NdotH_unsat * 0.5 + 0.5, _Roughness ) ).r;
                
               
                
                //////////////
                // beckmann distribution function
                float _RoughnessSqr = _Roughness * _Roughness;
                float NdotHSqr = NdotH * NdotH;
		        float r1 = 1.0 / ( 3.14 * _RoughnessSqr * NdotHSqr * NdotHSqr);
		        float r2 = (NdotHSqr - 1.0) / (_RoughnessSqr * NdotHSqr);
		        float roughness_beckmann = r1 * exp(r2);
		        
		        
		        //roughness_beckmann = saturate(roughness_beckmann);
                //
                
                //////////////
                // Blinn's NDF
                //float alpha = acos(NdotH);
				//float roughness_Blinn = _GaussConstant*exp(-(alpha*alpha)/(_Roughness * _Roughness));
                
                
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
                float fresnel = pow( 1.0 - VdotH, 5.0 );
                fresnel *= ( 1.0 - _Fresnel );
                fresnel += _Fresnel;
 
 				float3 Cspec = lerp(float3(0.3,0.3,0.3), s.Albedo, _Fresnel);
                float3 spec = float3 ( (fresnel * geo * roughness_beckmann ) / ( NdotV * NdotL + 0.02)); // 分母的pi一般被认为是包含到light参数里面去了所以这里不算
				//spec  *= Cspec;
				

 				float3 diff = (1 - _Fresnel) * s.Albedo;
 
 
				//
				//float3 refDirW =  normalize( reflect(viewDir, s.Normal));
				//float3 envRadiance = texCUBE(_Cube, refDirW);

				//float4 refDirW;
				//refDirW.xyz = normalize( reflect(viewDir, s.Normal));
				//refDirW.w = roughness_beckmann * 8;;
				//float3 envRadiance = texCUBEbias(_Cube, refDirW);

				//float3 reflColor = s.Emission;

                fixed4 c;

				fixed3 lightColor;
#ifdef USE_REAL_LIGHT
				lightColor = _LightColor0.rgb;
#else
				lightColor = _LightColor.rgb;
#endif


                //c.rgb = NdotL * (lightColor.rgb * atten + envRadiance * _EnvLightIntensity)  * ( spec +  diff);
                
                
                // AO
                //fixed3 aoForDiff = ShadeSH9 (float4(s.Normal, 1.0));
                //fixed3 aoForSpec = ShadeSH9 (float4(r, 1.0));
                
 
               	c.rgb = NdotL * (lightColor.rgb * atten)  * spec * Cspec;
				//c.rgb += NdotL * envRadiance * Cspec * _EnvLightIntensity *  _Fresnel;
               	c.rgb += NdotL * (lightColor.rgb * atten)  * diff;
                 
                
                c.a = s.Alpha;
                 
                
                //return float4(aoForSpec,1);
                //return float4(aoForDiff,1);
                
				//return  float4(envRadiance * _EnvLightIntensity, 1);
                return c;
                //return roughness;
				//return float4(s.Normal, 1);
            }
            
           v2f_surf vert (appdata_full v) 
           {
       			v2f_surf o;

				o.pos = mul (UNITY_MATRIX_MVP, v.vertex);
				o.pack0.xy = TRANSFORM_TEX(v.texcoord, _MainTex);

				o.lmap.xy = v.texcoord1.xy * unity_LightmapST.xy + unity_LightmapST.zw;

				float3 worldN = mul((float3x3)_Object2World, SCALED_NORMAL);

				o.viewDir = WorldSpaceViewDir(v.vertex);
				
				float3 binormal = cross(v.normal, v.tangent.xyz) * v.tangent.w;
				float3x3 rotation = float3x3(v.tangent.xyz, binormal, v.normal);
				o.TtoW0 = mul(rotation, ((float3x3)_Object2World)[0].xyz)*unity_Scale.w;
				o.TtoW1 = mul(rotation, ((float3x3)_Object2World)[1].xyz)*unity_Scale.w;
				o.TtoW2 = mul(rotation, ((float3x3)_Object2World)[2].xyz)*unity_Scale.w;
	
#ifdef USE_REAL_LIGHT
				o.lightDir = WorldSpaceLightDir(v.vertex);
#else
				o.lightDir = _WorldSpaceLightDir.xyz;
#endif


				// SH/ambient and vertex lights

				float3 shlight = ShadeSH9 (float4(worldN,1.0));
				o.vlight = shlight;
				#ifdef VERTEXLIGHT_ON
				float3 worldPos = mul(_Object2World, v.vertex).xyz;
				o.vlight += Shade4PointLights (
				unity_4LightPosX0, unity_4LightPosY0, unity_4LightPosZ0,
				unity_LightColor[0].rgb, unity_LightColor[1].rgb, unity_LightColor[2].rgb, unity_LightColor[3].rgb,
				unity_4LightAtten0, worldPos, worldN );
				#endif // VERTEXLIGHT_ON


				// pass lighting information to pixel shader
				TRANSFER_VERTEX_TO_FRAGMENT(o);
				return o;
			}
            
			fixed4 frag (v2f_surf IN) : SV_Target 
			{
				Input surfIN = (Input)0;
				surfIN.uv_MainTex = IN.pack0.xy;
				surfIN.TtoW0 = IN.TtoW0;
				surfIN.TtoW1 = IN.TtoW1;
				surfIN.TtoW2 = IN.TtoW2;
				
				SurfaceOutput o = (SurfaceOutput)0;
				o.Albedo = 0.0;
				o.Emission = 0.0;
				o.Specular = 0.0;
				o.Alpha = 0.0;
				o.Gloss = 0.0;

				// call surface function
				surf (surfIN, o);

				// compute lighting & shadowing factor
				fixed atten = LIGHT_ATTENUATION(IN);
				fixed4 c = 0;

				// realtime lighting: call lighting function

				c = LightingCookTorrance (o, IN.lightDir, normalize(half3(IN.viewDir)), atten);
				
				

				c.rgb += o.Albedo * IN.vlight;


				// lightmaps:


				// single lightmap
				fixed4 lmtex = tex2D(unity_Lightmap, IN.lmap.xy);
				fixed3 lm = DecodeLightmap (lmtex);


				// combine lightmaps with realtime shadows
				#ifdef SHADOWS_SCREEN
				#if defined(UNITY_NO_RGBM)
				c.rgb += o.Albedo * min(lm, atten*2);
				#else
				c.rgb += o.Albedo * max(min(lm,(atten*2)*lmtex.rgb), lm*atten);
				#endif
				#else // SHADOWS_SCREEN
				c.rgb += o.Albedo * lm * _LightMapModifier;
				#endif
				c.a = o.Alpha;


				return c;
			}

            ENDCG
        }
    }
FallBack "Mobile/Diffuse"
}